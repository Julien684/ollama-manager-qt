//! Ollama Manager — GUI Qt6 (via `qtrs`), binaire unique.
//!
//! Télécharge, met à jour et supprime des modèles Ollama. L'adresse du
//! serveur est configurable au lancement et persistée (QSettings).
//!
//! Modèle d'ownership qtrs (pattern validé sur solus-external-kernel-manager) :
//!  - La fenêtre (MainWindow) est dans `Rc<RefCell<..>>` (clonable dans les
//!    callbacks).
//!  - Les widgets « interactifs » (table, barre, statut) sont gardés en
//!    `Rc<RefCell<..>>`, marqués `set_has_parent()`, puis enveloppés dans des
//!    `ScrollArea` via `set_widget(&dyn AsWidget)`.
//!  - Les widgets « statiques » (boutons, groupboxes, line edits, menu) sont
//!    parentés puis transférés aux layouts (`.add()` les consomme).
//!  - `Layouts` retient en vie les layouts (un layout dropé supprimerait son
//!    C++ et orphelinerait ses enfants).
//!  - Travail long (HTTP) → `std::thread` + `mpsc` ; un `qtrs::Timer` (200 ms)
//!    fait la « pompe » file → widgets dans le thread UI.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;

use qtrs::prelude::*;
use qtrs::{Application, MessageBox, Timer};

mod ollama;
use ollama::{fmt_bytes, fmt_date, delete_model, get_version, list_models,
             pull_model, Model};

// ------------------------------------------------------------------
// Messages worker → UI
// ------------------------------------------------------------------
enum WorkerMsg {
    /// Résultats d'un `GET /api/tags` : (série JSON des modèles, message d'erreur si échec).
    List(Option<String>, String),
    /// Progression d'un pull : (nom, pct, statut).
    Progress { name: String, pct: u32, status: String },
    /// Fin d'une action (pull / delete) : (nom, ok, message).
    ActionDone { name: String, ok: bool, message: String },
}

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

const SETTINGS_ORG: &str = "OllamaManager";
const SETTINGS_APP: &str = "ollama-manager-qt";

/// Chemin du fichier de configuration (~/.config/OllamaManager/...).
///
/// On ne passe pas par `qtrs::Settings` pour la **lecture** : son chemin
/// de read (prebuilt C++) renvoie la chaîne vide même quand le `.ini` est
/// écrit par Qt lui-même (persistance de l'URL cassée). On gère donc le
/// fichier INI directement — format identique à celui de QSettings,
/// donc rétro-compatible.
fn config_path() -> std::path::PathBuf {
    let base = std::env::var("XDG_CONFIG_HOME")
        .ok()
        .filter(|p| !p.is_empty())
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var("HOME").ok().map(std::path::PathBuf::from))
        .unwrap_or_else(std::env::temp_dir);
    base.join(".config").join(SETTINGS_ORG).join(format!("{SETTINGS_APP}.ini"))
}

/// Parse la clé `serverUrl` d'un contenu INI Qt. Testable sans GUI.
pub fn parse_ini_server_url(content: &str) -> Option<String> {
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("serverUrl=") {
            let v = rest.trim();
            // QSettings encadre les valeurs contenant `:`, `/` ou espaces
            // dans des guillemets « ... ».
            if v.len() >= 2 && v.starts_with('"') && v.ends_with('"') {
                return Some(v[1..v.len() - 1].to_string());
            }
            return Some(v.to_string());
        }
    }
    None
}

fn load_server_url() -> String {
    std::fs::read_to_string(config_path())
        .ok()
        .and_then(|c| parse_ini_server_url(&c))
        .unwrap_or_default()
}

fn save_server_url(url: &str) {
    let path = config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Échappement minimal des caractères spéciaux QSettings.
    let escaped = url.replace('"', "\\\"").replace('\\', "\\\\");
    let need_escape = url.contains(&['"', '\\', ' ', ':', '['][..]);
    let val = if need_escape { format!("\"{escaped}\"") } else { escaped };
    let _ = std::fs::write(&path, format!("[General]\nserverUrl={val}\n"));
}

/// Lit le texte courant d'un `QLineEdit` via son pointeur C++.
fn read_line_edit(ptr: *mut qtrs::ffi::QLineEdit) -> String {
    unsafe { qtrs::ffi::QLineEdit_text(ptr) }
}

/// Enveloppe un widget interactif dans un `ScrollArea` (pattern qtrs).
fn wrap_scroll(w: &Rc<RefCell<impl qtrs::AsWidget>>, parent: &dyn AsWidget) -> ScrollArea {
    let sa = ScrollArea::new().parent(parent).set_widget_resizable(true).build();
    sa.set_widget(&*w.borrow());
    sa
}

/// Boîte de confirmation modale (Oui/Non) parentée à la fenêtre.
fn confirm(parent: &dyn AsWidget, text: &str) -> bool {
    let mb = MessageBox::new()
        .icon(3) // Question
        .text(text)
        .window_title("Ollama Manager")
        .standard_buttons(0x4000 | 0x400) // Yes | No
        .parent(parent)
        .build();
    mb.exec() == 0x4000 // QMessageBox::Yes
}

// ------------------------------------------------------------------
// Handles partagés entre la construction et les callbacks
// ------------------------------------------------------------------
struct Ui {
    table: Rc<RefCell<TableWidget>>,
    bar: Rc<RefCell<ProgressBar>>,
    status: Rc<RefCell<Label>>,
    /// Modèles courants (index de ligne = position dans ce Vec).
    models: Rc<RefCell<Vec<Model>>>,
}

struct Layouts {
    #[allow(dead_code)]
    root: VBoxLayout,
    #[allow(dead_code)]
    srv: HBoxLayout,
    #[allow(dead_code)]
    dl: HBoxLayout,
    #[allow(dead_code)]
    actions: HBoxLayout,
    #[allow(dead_code)]
    modbox: VBoxLayout,
    #[allow(dead_code)]
    bar_row: HBoxLayout,
    /// Widget central sans parent : SANS lui, qtrs supprime le C++ du widget
    /// (donc toute la sous-arborescence) à la fin de `build_ui` → segfault dans
    /// `app.exec()`. Doit vivre aussi longtemps que la fenêtre.
    #[allow(dead_code)]
    central: Widget,
}

// ------------------------------------------------------------------
// Actions (utilisées par les boutons ET par le timer)
// ------------------------------------------------------------------

/// Remplit la table avec la liste de modèles reçue.
fn fill_table(ui: &Ui, models: Vec<Model>) {
    ui.models.borrow_mut().clone_from(&models);
    let t = ui.table.borrow();
    t.set_row_count(models.len() as i32);
    for (i, m) in models.iter().enumerate() {
        t.set_item(i as i32, 0, &m.name);
        t.set_item(i as i32, 1, &fmt_bytes(m.size));
        let meta = format!("{} {}", m.params, m.quant).trim().to_string();
        t.set_item(i as i32, 2, &meta);
        t.set_item(i as i32, 3, &fmt_date(&m.modified_at));
        t.set_item(i as i32, 4, &m.caps);
    }
}

/// Lance un pull dans un thread, avec remontée de progression via `tx`.
fn start_pull(
    tx: mpsc::Sender<WorkerMsg>,
    status: Rc<RefCell<Label>>,
    name: String,
    url: &str,
) {
    status.borrow_mut().set_text(&format!("[{name}] démarrage…"));
    let name0 = name.clone();
    let url0 = url.to_string();
    std::thread::spawn(move || {
        let res = pull_model(&url0, &name0, |pct, st| {
            let _ = tx.send(WorkerMsg::Progress {
                name: name0.clone(),
                pct,
                status: st.to_string(),
            });
        });
        let (ok, message) = match res {
            Ok(()) => (true, "téléchargé / mis à jour".to_string()),
            Err(e) => (false, e.to_string()),
        };
        let _ = tx.send(WorkerMsg::ActionDone {
            name: name0,
            ok,
            message,
        });
    });
}

/// Lance une lecture de la liste des modèles dans un thread.
fn refresh_list(
    tx: &mpsc::Sender<WorkerMsg>,
    url: &str,
    status: &Rc<RefCell<Label>>,
) {
    status.borrow_mut().set_text("Interrogation d'Ollama…");
    let tx = tx.clone();
    let url = url.to_string();
    std::thread::spawn(move || {
        let (payload, err) = match list_models(&url) {
            Ok(m) => (
                Some(serde_json::to_string(&m).unwrap_or_default()),
                String::new(),
            ),
            Err(e) => (None, e.to_string()),
        };
        let _ = tx.send(WorkerMsg::List(payload, err));
    });
}

// ------------------------------------------------------------------
// Construction de l'interface
// ------------------------------------------------------------------
fn build_ui(
    window: &Rc<RefCell<MainWindow>>,
    tx: mpsc::Sender<WorkerMsg>,
) -> (Ui, Layouts) {
    // ---- Widgets communs (en Rc, sans parent) ------------------------------
    let table = Rc::new(RefCell::new(TableWidget::new(0, 5).build()));
    let bar = Rc::new(RefCell::new(
        ProgressBar::new().range(0, 100).value(0).format("%p %").build(),
    ));
    let status = Rc::new(RefCell::new(Label::new("En attente…").build()));
    let models = Rc::new(RefCell::new(Vec::new()));

    table.borrow_mut().set_has_parent();
    bar.borrow_mut().set_has_parent();
    status.borrow_mut().set_has_parent();
    table.borrow_mut().set_horizontal_header_labels(&[
        "Modèle", "Taille", "Paramètres", "Modifié", "Capacités",
    ]);
    table.borrow_mut().set_column_width(0, 380);
    table.borrow_mut().set_column_width(1, 80);
    table.borrow_mut().set_column_width(2, 120);
    table.borrow_mut().set_column_width(3, 90);
    table.borrow_mut().set_column_width(4, 260);

    let w = window.borrow();
    let wp: &dyn AsWidget = &*w;
    let ui = Ui {
        table: table.clone(),
        bar: bar.clone(),
        status: status.clone(),
        models: models.clone(),
    };

    // ---- Menu ---------------------------------------------------------------
    let file_menu = Menu::new("Fichier")
        .action("Quitter", || std::process::exit(0))
        .parent(wp)
        .build();
    let menubar = MenuBar::new()
        .add_menu(file_menu)
        .parent(wp)
        .build();
    w.set_menu_bar(&menubar);

    // ---- Widget central + layout racine ------------------------------------
    let central = Widget::new().build();
    let mut root = VBoxLayout::with_parent(&central);
    root.set_contents_margins(12, 12, 12, 12);
    root.set_spacing(10);

    // ---- Ligne serveur : [ QLineEdit | Tester | Enregistrer ] --------------
    // Titre du champ : sans lui l'utilisateur ne sait pas quoi y saisir.
    let srv_title =
        Label::new("Serveur Ollama (port par défaut : 11434)")
            .parent(&central)
            .build();
    root.add(srv_title);
    let srv_row = Widget::new().parent(&central).build();
    let mut srv_l = HBoxLayout::with_parent(&srv_row);
    srv_l.set_spacing(8);

    let url_edit = LineEdit::new(load_server_url()).parent(&srv_row).build();
    // Pointeur C++ capté avant transfert au layout (le widget est consommé).
    let url_ptr = url_edit.widget_ptr() as *mut qtrs::ffi::QLineEdit;
    srv_l.add(url_edit);

    // Bouton « Tester » : lit le champ via le pointeur C++.
    let test_btn = PushButton::new("Tester")
        .on_clicked({
            let status = status.clone();
            move || {
                let url = read_line_edit(url_ptr).trim().to_string();
                if url.is_empty() {
                    status.borrow_mut().set_text("Saisissez une adresse de serveur.");
                    return;
                }
                status.borrow_mut().set_text("Test de connexion…");
                match get_version(&url) {
                    Ok(v) => status
                        .borrow_mut()
                        .set_text(&format!("Connecté — Ollama version {v}")),
                    Err(e) => status
                        .borrow_mut()
                        .set_text(&format!("Échec de connexion : {e}")),
                }
            }
        })
        .parent(&srv_row)
        .build();
    srv_l.add(test_btn);

    // Bouton « Enregistrer » : persiste l'adresse puis recharge la liste.
    let save_btn = PushButton::new("Enregistrer")
        .on_clicked({
            let status = status.clone();
            let tx = tx.clone();
            move || {
                let url = read_line_edit(url_ptr).trim().to_string();
                if url.is_empty() {
                    status.borrow_mut().set_text("Saisissez une adresse de serveur.");
                    return;
                }
                save_server_url(&url);
                status.borrow_mut()
                    .set_text(&format!("Adresse enregistrée : {url}"));
                refresh_list(&tx, &url, &status);
            }
        })
        .parent(&srv_row)
        .build();
    srv_l.add(save_btn);

    root.add(srv_row);

    // ---- Télécharger un modèle ----------------------------------------------
    let dl_box =
        GroupBox::new("Télécharger un modèle").parent(&central).build();
    // Layout VERTICAL pour le GroupBox (c'était ce qui manquait : le dl_row
    // était enfant du GroupBox mais sans layout parent, donc invisible).
    let mut dl_outer = VBoxLayout::with_parent(&dl_box);
    dl_outer.set_spacing(8);

    let dl_row = Widget::new().parent(&dl_box).build();
    let mut dl_l = HBoxLayout::with_parent(&dl_row);
    dl_l.set_spacing(8);

    // Placeholder : aide l'utilisateur à saisir le nom du modèle.
    let pull_edit = LineEdit::new("ex : llama3.2:3b").parent(&dl_row).build();
    let pull_ptr = pull_edit.widget_ptr() as *mut qtrs::ffi::QLineEdit;
    dl_l.add(pull_edit);

    let download_btn = PushButton::new("Télécharger")
        .on_clicked({
            let tx = tx.clone();
            let status = status.clone();
            move || {
                let name = read_line_edit(pull_ptr).trim().to_string();
                if name.is_empty() {
                    status.borrow_mut().set_text("Entrez un nom de modèle.");
                    return;
                }
                let url = load_server_url();
                if url.trim().is_empty() {
                    status
                        .borrow_mut()
                        .set_text("Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                start_pull(tx.clone(), status.clone(), name, &url);
            }
        })
        .parent(&dl_row)
        .build();
    dl_l.add(download_btn);
    dl_outer.add(dl_row);
    root.add(dl_box);

    // ---- Modèles locaux ------------------------------------------------------
    let mod_box =
        GroupBox::new("Modèles locaux").parent(&central).build();
    let mut mod_l = VBoxLayout::with_parent(&mod_box);
    mod_l.set_spacing(8);

    let actions_row = Widget::new().parent(&mod_box).build();
    let mut actions_l = HBoxLayout::with_parent(&actions_row);
    actions_l.set_spacing(8);

    // Actualiser
    let refresh_btn = PushButton::new("Actualiser")
        .on_clicked({
            let tx = tx.clone();
            let status = status.clone();
            move || {
                let url = load_server_url();
                if url.trim().is_empty() {
                    status
                        .borrow_mut()
                        .set_text("Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                refresh_list(&tx, &url, &status);
            }
        })
        .parent(&actions_row)
        .build();
    actions_l.add(refresh_btn);

    // Mettre à jour (ligne sélectionnée)
    let update_btn = PushButton::new("Mettre à jour")
        .on_clicked({
            let tx = tx.clone();
            let status = status.clone();
            let models = models.clone();
            let table_ref = table.clone();
            move || {
                let row = table_ref.borrow().current_row();
                if row < 0 {
                    status
                        .borrow_mut()
                        .set_text("Sélectionnez d'abord un modèle à mettre à jour.");
                    return;
                }
                let Some(name) = models.borrow().get(row as usize).map(|m| m.name.clone()) else {
                    status.borrow_mut().set_text("Ligne invalide.");
                    return;
                };
                let url = load_server_url();
                if url.trim().is_empty() {
                    status
                        .borrow_mut()
                        .set_text("Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                start_pull(tx.clone(), status.clone(), name, &url);
            }
        })
        .parent(&actions_row)
        .build();
    actions_l.add(update_btn);

    // Tout mettre à jour (séquencé dans un thread)
    let update_all_btn = PushButton::new("Tout mettre à jour")
        .on_clicked({
            let tx = tx.clone();
            let status = status.clone();
            let models = models.clone();
            move || {
                let names: Vec<String> =
                    models.borrow().iter().map(|m| m.name.clone()).collect();
                if names.is_empty() {
                    status
                        .borrow_mut()
                        .set_text("Aucun modèle local à mettre à jour.");
                    return;
                }
                let url = load_server_url();
                if url.trim().is_empty() {
                    status
                        .borrow_mut()
                        .set_text("Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                status
                    .borrow_mut()
                    .set_text(&format!(
                        "Mise à jour de {} modèle(s)…",
                        names.len()
                    ));
                let tx2 = tx.clone();
                let url2 = url.clone();
                std::thread::spawn(move || {
                    for name in &names {
                        let _ = pull_model(&url2, name, |pct, st| {
                            let _ = tx2.send(WorkerMsg::Progress {
                                name: name.clone(),
                                pct,
                                status: st.to_string(),
                            });
                        });
                        let _ = tx2.send(WorkerMsg::Progress {
                            name: name.clone(),
                            pct: 100,
                            status: "terminé".into(),
                        });
                    }
                    let _ = tx2.send(WorkerMsg::ActionDone {
                        name: String::new(),
                        ok: true,
                        message: "toutes les mises à jour sont terminées".into(),
                    });
                });
            }
        })
        .parent(&actions_row)
        .build();
    actions_l.add(update_all_btn);

    // Supprimer (ligne sélectionnée)
    let delete_btn = PushButton::new("Supprimer")
        .on_clicked({
            let tx = tx.clone();
            let status = status.clone();
            let models = models.clone();
            let table_ref = table.clone();
            let window = Rc::clone(window);
            move || {
                let row = table_ref.borrow().current_row();
                if row < 0 {
                    status
                        .borrow_mut()
                        .set_text("Sélectionnez d'abord un modèle à supprimer.");
                    return;
                }
                let Some(name) = models.borrow().get(row as usize).map(|m| m.name.clone()) else {
                    status.borrow_mut().set_text("Ligne invalide.");
                    return;
                };
                let w = window.borrow();
                if !confirm(&*w, &format!("Supprimer « {name} » ?")) {
                    return;
                }
                let url = load_server_url();
                let tw = tx.clone();
                std::thread::spawn(move || {
                    let (ok, message) = match delete_model(&url, &name) {
                        Ok(()) => (true, "supprimé".to_string()),
                        Err(e) => (false, e.to_string()),
                    };
                    let _ = tw.send(WorkerMsg::ActionDone {
                        name,
                        ok,
                        message,
                    });
                });
            }
        })
        .parent(&actions_row)
        .build();
    actions_l.add(delete_btn);

    mod_l.add(actions_row);
    mod_l.add(wrap_scroll(&table, wp));
    root.add(mod_box);

    // ---- Barre de progression + statut --------------------------------------
    let bar_row = Widget::new().parent(&central).build();
    let mut bar_l = HBoxLayout::with_parent(&bar_row);
    bar_l.set_spacing(10);
    bar_l.add(wrap_scroll(&bar, wp));
    bar_l.add(wrap_scroll(&status, wp));
    root.add(bar_row);

    // Enveloppe défilante.
    let central_scroll =
        ScrollArea::new().parent(wp).set_widget_resizable(true).build();
    central_scroll.set_widget(&central);
    w.set_central_widget(&central_scroll);

    let layouts = Layouts {
        root,
        srv: srv_l,
        dl: dl_l,
        actions: actions_l,
        modbox: mod_l,
        bar_row: bar_l,
        central,
    };

    (ui, layouts)
}

// ------------------------------------------------------------------
fn main() {
    let app = Application::new();
    let (tx, rx) = mpsc::channel::<WorkerMsg>();

    let window = Rc::new(RefCell::new(
        MainWindow::new()
            .window_title("Ollama Manager")
            .size(1020, 640)
            .build(),
    ));

    // Ordre de drop CRITIQUE à la fermeture de l'application : `ui`
    // (table / barre / statut — enfants de `central`) doit être dropé **avant**
    // `_layouts` (qui contient `central`). Si `central` était supprimé en C++
    // en premier, le drop de `ui` appellerait `disconnectAll` sur des pointeurs
    // morts → SIGSEGV (logs « wildcard call disconnects from destroyed
    // signal »). On lie donc `ui` *en dernier* : le drop Rust est l'inverse
    // de l'ordre de déclaration, `ui` passe donc au premier rang du drop.
    let _ui_layouts = build_ui(&window, tx.clone());
    let _layouts = _ui_layouts.1;
    let ui = _ui_layouts.0;
    window.borrow().show();

    // Chargement initial si une adresse est déjà configurée.
    {
        let url = load_server_url();
        if !url.trim().is_empty() {
            refresh_list(&tx, &url, &ui.status);
        } else {
            ui.status
                .borrow_mut()
                .set_text("Saisissez et enregistrez l'adresse du serveur Ollama.");
        }
    }

    // ---- Timer « pompe » : file worker → widgets ---------------------------
    // Le timer est déclaré au niveau de `main` (pas dans un bloc `{ }`) : la
    // variable doit vivre jusqu'à la fin de `app.exec()`, sinon il serait
    // détruit avant le démarrage de la boucle d'événements et les résultats
    // des workers ne seraient jamais pompés vers l'interface.
    let tx2 = tx.clone();
    let _timer = Timer::new(200)
        .on_timeout({
            let table = Rc::clone(&ui.table);
            let bar = Rc::clone(&ui.bar);
            let status = Rc::clone(&ui.status);
            let models = Rc::clone(&ui.models);
            move || {
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        WorkerMsg::List(Some(payload), _) => {
                            let parsed: Vec<Model> =
                                serde_json::from_str(&payload).unwrap_or_default();
                            let count = parsed.len();
                            let ui_l = Ui {
                                table: table.clone(),
                                bar: bar.clone(),
                                status: status.clone(),
                                models: models.clone(),
                            };
                            fill_table(&ui_l, parsed);
                            status
                                .borrow_mut()
                                .set_text(&format!(
                                    "{} modèle(s) chargé(s).",
                                    count
                                ));
                        }
                        WorkerMsg::List(None, err) => {
                            status
                                .borrow_mut()
                                .set_text(&format!("Erreur : {err}"));
                        }
                        WorkerMsg::Progress { name, pct, status: st } => {
                            bar.borrow_mut().set_value(pct as i32);
                            status
                                .borrow_mut()
                                .set_text(&format!("[{name}] {st} — {pct}%"));
                        }
                        WorkerMsg::ActionDone { name, ok, message } => {
                            let mb = MessageBox::new()
                                .icon(if ok { 2 } else { 0 })
                                .text(if ok {
                                    "Terminé"
                                } else {
                                    "Échec de l'opération"
                                })
                                .informative_text(&format!("{name} : {message}"))
                                .window_title("Ollama Manager")
                                .standard_buttons(0x400)
                                .build();
                            let _ = mb.exec();
                            bar.borrow_mut().set_value(0);
                            // Recharger la liste après une action réussie.
                            let url = load_server_url();
                            if ok && !url.trim().is_empty() {
                                let tx3 = tx2.clone();
                                let url3 = url.clone();
                                std::thread::spawn(move || {
                                    let (payload, err) = match list_models(&url3) {
                                        Ok(m) => (
                                            Some(
                                                serde_json::to_string(&m)
                                                    .unwrap_or_default(),
                                            ),
                                            String::new(),
                                        ),
                                        Err(e) => (None, e.to_string()),
                                    };
                                    let _ = tx3.send(WorkerMsg::List(payload, err));
                                });
                            }
                        }
                    }
                }
            }
        })
        .build();

    app.exec();
}

// ------------------------------------------------------------------
// Tests unitaires (logique de persistance, sans GUI)
// ------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ini_url_simple() {
        assert_eq!(
            parse_ini_server_url("[General]\nserverUrl=http://192.168.122.1:11434\n"),
            Some("http://192.168.122.1:11434".to_string())
        );
    }

    #[test]
    fn ini_url_guillemets() {
        // QSettings encadre les valeurs contenant `:`, `/`…
        assert_eq!(
            parse_ini_server_url("[General]\nserverUrl=\"http://192.168.122.1:11434\"\n"),
            Some("http://192.168.122.1:11434".to_string())
        );
    }

    #[test]
    fn ini_url_absent() {
        assert_eq!(parse_ini_server_url("[General]\nother=1\n"), None);
    }

    #[test]
    fn ini_url_vide() {
        assert_eq!(parse_ini_server_url(""), None);
    }

    #[test]
    fn ini_url_ignore_autres_cles() {
        assert_eq!(
            parse_ini_server_url("[General]\nfoo=bar\nserverUrl=http://localhost:11434\n"),
            Some("http://localhost:11434".to_string())
        );
    }
}
