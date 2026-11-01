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
use std::rc::{Rc, Weak};
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

/// Parse une clé quelconque d'un contenu INI Qt. Testable sans GUI.
pub fn parse_ini_key(content: &str, key: &str) -> Option<String> {
    let prefix = format!("{key}=");
    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(&prefix) {
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

/// Parse la clé `serverUrl` d'un contenu INI Qt. Testable sans GUI.
pub fn parse_ini_server_url(content: &str) -> Option<String> {
    parse_ini_key(content, "serverUrl")
}

/// Parse la clé `apiKey` (clé API facultative) d'un contenu INI Qt.
pub fn parse_ini_api_key(content: &str) -> Option<String> {
    parse_ini_key(content, "apiKey")
}

/// Lit (URL du serveur, clé API) depuis la configuration.
fn load_config() -> (String, String) {
    let content = std::fs::read_to_string(config_path()).unwrap_or_default();
    (
        parse_ini_server_url(&content).unwrap_or_default(),
        parse_ini_api_key(&content).unwrap_or_default(),
    )
}

fn load_server_url() -> String {
    load_config().0
}

/// Échappe une valeur pour un fichier INI façon QSettings.
fn ini_value(raw: &str) -> String {
    let escaped = raw.replace('"', "\\\"").replace('\\', "\\\\");
    let need_escape = raw.contains(&['"', '\\', ' ', ':', '['][..]);
    if need_escape {
        format!("\"{escaped}\"")
    } else {
        escaped
    }
}

/// Écrit l'URL ET la clé API (une clé vide retire la ligne `apiKey`).
fn save_config(url: &str, api_key: &str) {
    let path = config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut out = String::from("[General]\n");
    out.push_str(&format!("serverUrl={}\n", ini_value(url)));
    if !api_key.trim().is_empty() {
        out.push_str(&format!("apiKey={}\n", ini_value(api_key.trim())));
    }
    let _ = std::fs::write(&path, out);
}

/// Applique au client HTTP la clé API saisie dans un champ (vide = aucune).
fn apply_api_key_from(ptr: *mut qtrs::ffi::QLineEdit) {
    let key = read_line_edit(ptr);
    ollama::set_api_key(Some(key));
}

/// Lit le texte courant d'un `QLineEdit` via son pointeur C++.
fn read_line_edit(ptr: *mut qtrs::ffi::QLineEdit) -> String {
    unsafe { qtrs::ffi::QLineEdit_text(ptr) }
}


/// Ecrit dans la barre de statut si le widget existe encore.
/// Les callbacks capturent des `Weak` (jamais des `Rc`) : ils ne doivent pas
/// prolonger la vie des widgets, sinon leurs wrappers ne sont detruits
/// qu'APRES que Qt a libere les objets C++ -> « QObject::disconnect:
/// wildcard call disconnects from destroyed signal » a la fermeture.
fn set_status(w: &Weak<RefCell<Label>>, msg: &str) {
    if let Some(s) = w.upgrade() {
        s.borrow_mut().set_text(msg);
    }
}

/// Ligne selectionnee du tableau (-1 si le widget n'existe plus).
fn table_row(w: &Weak<RefCell<TableWidget>>) -> i32 {
    w.upgrade().map(|t| t.borrow().current_row()).unwrap_or(-1)
}

/// Remet la barre de progression a zero si elle existe encore.
fn reset_bar(w: &Weak<RefCell<ProgressBar>>) {
    if let Some(b) = w.upgrade() {
        b.borrow_mut().set_value(0);
    }
}

/// Cherche l'icone de l'application (installee d'abord, puis locale au build).
fn app_icon_path() -> Option<String> {
    for p in [
        "/usr/share/pixmaps/ollama-manager-qt.png",
        "/usr/share/icons/hicolor/256x256/apps/ollama-manager-qt.png",
        "icons/ollama-manager-qt-256.png",
    ] {
        if std::path::Path::new(p).exists() {
            return Some(p.to_string());
        }
    }
    None
}

/// Enveloppe un widget interactif dans un `ScrollArea` (pattern qtrs).
fn wrap_scroll(w: &Rc<RefCell<impl qtrs::AsWidget>>, parent: &dyn AsWidget) -> ScrollArea {
    let sa = ScrollArea::new().parent(parent).set_widget_resizable(true).build();
    sa.set_widget(&*w.borrow());
    sa
}

/// Boutons Qt utilisés pour les dialogues (valeurs QMessageBox::StandardButton).
/// ATTENTION : `Ok` vaut 0x400 et `Yes` 0x4000 — les confondre fait qu'un clic
/// sur « Ok » renvoie une valeur que le code ne reconnaît pas (action annulée
/// en silence, sans erreur).
pub const BTN_YES: i32 = 0x0000_4000;
pub const BTN_CANCEL: i32 = 0x0040_0000;

/// Boîte de confirmation modale (Oui / Annuler).
///
/// Sans parent : voir la note sur le double emprunt de `RefCell` ci-dessus.
/// « Annuler » plutôt que « Ok »/« Non » : deux boutons validants (Oui + Ok)
/// prêtaient à confusion.
fn confirm(text: &str) -> bool {
    let mb = MessageBox::new()
        .icon(3) // Question
        .text(text)
        .window_title("Ollama Manager")
        .standard_buttons(BTN_YES | BTN_CANCEL)
        // Bouton par defaut explicite : sans cela, selon le style Qt, la touche
        // Entree peut activer « Annuler » et annuler la suppression.
        .default_button(BTN_YES)
        .build();
    mb.exec() == BTN_YES
}

/// Nom du modèle situé à la ligne `row` (-1 = aucune sélection).
/// Fonction pure, testable sans interface.
pub fn model_name_at(models: &[Model], row: i32) -> Option<String> {
    if row < 0 {
        return None;
    }
    models.get(row as usize).map(|m| m.name.clone())
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
    api: HBoxLayout,
    #[allow(dead_code)]
    dl: GridLayout,
    #[allow(dead_code)]
    actions: HBoxLayout,
    #[allow(dead_code)]
    modbox: VBoxLayout,
    #[allow(dead_code)]
    bar_row: VBoxLayout,
    #[allow(dead_code)]
    log_row: VBoxLayout,
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
    status: Weak<RefCell<Label>>,
    name: String,
    url: &str,
) {
    set_status(&status, &format!("[{name}] démarrage…"));
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
    status: &Weak<RefCell<Label>>,
) {
    set_status(status, "Interrogation d'Ollama…");
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

    // ---- Clé API facultative (créée tôt pour être lue par « Tester ») ------
    let api_title = Label::new(
        "Clé API (facultative — laisser vide si le serveur n'en exige pas)",
    )
    .parent(&central)
    .build();
    let api_row = Widget::new().parent(&central).build();
    let mut api_l = HBoxLayout::with_parent(&api_row);
    api_l.set_spacing(8);
    let saved_key = load_config().1;
    let api_edit = LineEdit::new(&saved_key).parent(&api_row).build();
    api_edit.set_echo_mode(2); // 2 = Password : la clé n'est pas affichée
    let api_ptr = api_edit.widget_ptr() as *mut qtrs::ffi::QLineEdit;
    api_l.add(api_edit);

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
            let status = Rc::downgrade(&status);
            let api_p = api_ptr;
            move || {
                let url = read_line_edit(url_ptr).trim().to_string();
                apply_api_key_from(api_p);
                if url.is_empty() {
                    set_status(&status, "Saisissez une adresse de serveur.");
                    return;
                }
                set_status(&status, "Test de connexion…");
                match get_version(&url) {
                    Ok(v) =>
                        set_status(&status, &format!("Connecté — Ollama version {v}")),
                    Err(e) =>
                        set_status(&status, &format!("Échec de connexion : {e}")),
                }
            }
        })
        .parent(&srv_row)
        .build();
    srv_l.add(test_btn);

    // Bouton « Enregistrer » : persiste l'adresse puis recharge la liste.
    let save_btn = PushButton::new("Enregistrer")
        .on_clicked({
            let status = Rc::downgrade(&status);
            let tx = tx.clone();
            let api_p = api_ptr;
            move || {
                let url = read_line_edit(url_ptr).trim().to_string();
                if url.is_empty() {
                    set_status(&status, "Saisissez une adresse de serveur.");
                    return;
                }
                let key = read_line_edit(api_p);
                save_config(&url, &key);
                ollama::set_api_key(Some(key.clone()));
                let msg = if key.trim().is_empty() {
                    format!("Adresse enregistrée : {url} (sans clé API)")
                } else {
                    format!("Adresse et clé API enregistrées : {url}")
                };
                set_status(&status, &msg);
                refresh_list(&tx, &url, &status);
            }
        })
        .parent(&srv_row)
        .build();
    srv_l.add(save_btn);

    root.add(srv_row);
    root.add(api_title);
    root.add(api_row);

    // ---- Télécharger un modèle ----------------------------------------------
    let dl_box =
        GroupBox::new("Télécharger un modèle").parent(&central).build();
    // GridLayout pose directement sur le GroupBox : les enfants sont ajoutes au
    // grid lui-meme. Un `Widget` intermediaire (avec son propre HBoxLayout)
    // ajoute a la VBox du GroupBox ne s'affichait pas du tout (section vide).
    let mut dl_g = GridLayout::with_parent(&dl_box);

    let dl_title =
        Label::new("Nom du modèle à télécharger (ex : llama3.2:3b)")
            .parent(&dl_box)
            .build();
    dl_g.add(dl_title, 0, 0, 1, 2);

    // Placeholder : le champ doit montrer ce qu'on attend avant toute saisie.
    let pull_edit = LineEdit::new("ex : llama3.2:3b").parent(&dl_box).build();
    let pull_ptr = pull_edit.widget_ptr() as *mut qtrs::ffi::QLineEdit;
    dl_g.add(pull_edit, 1, 0, 1, 1);

    let download_btn = PushButton::new("Télécharger")
        .on_clicked({
            let tx = tx.clone();
            let status = Rc::downgrade(&status);
            let api_p = api_ptr;
            move || {
                let name = read_line_edit(pull_ptr).trim().to_string();
                apply_api_key_from(api_p);
                if name.is_empty() {
                    set_status(&status, "Entrez un nom de modèle.");
                    return;
                }
                let url = load_server_url();
                if url.trim().is_empty() {
                    set_status(&status, "Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                start_pull(tx.clone(), status.clone(), name, &url);
            }
        })
        .parent(&dl_box)
        .build();
    dl_g.add(download_btn, 1, 1, 1, 1);
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
            let status = Rc::downgrade(&status);
            move || {
                let url = load_server_url();
                if url.trim().is_empty() {
                    set_status(&status, "Enregistrez d'abord l'adresse du serveur.");
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
            let status = Rc::downgrade(&status);
            let models = models.clone();
            let table_ref = Rc::downgrade(&table);
            move || {
                let row = table_row(&table_ref);
                if row < 0 {
                    set_status(
                        &status,
                        "Sélectionnez d'abord un modèle à mettre à jour.",
                    );
                    return;
                }
                let Some(name) = models.borrow().get(row as usize).map(|m| m.name.clone()) else {
                    set_status(&status, "Ligne invalide.");
                    return;
                };
                let url = load_server_url();
                if url.trim().is_empty() {
                    set_status(&status, "Enregistrez d'abord l'adresse du serveur.");
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
            let status = Rc::downgrade(&status);
            let models = models.clone();
            move || {
                let names: Vec<String> =
                    models.borrow().iter().map(|m| m.name.clone()).collect();
                if names.is_empty() {
                    set_status(&status, "Aucun modèle local à mettre à jour.");
                    return;
                }
                let url = load_server_url();
                if url.trim().is_empty() {
                    set_status(&status, "Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                set_status(
                    &status,
                    &format!("Mise à jour de {} modèle(s)…", names.len()),
                );
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
            let status = Rc::downgrade(&status);
            let models = models.clone();
            let table_ref = Rc::downgrade(&table);
            move || {
                let row = table_row(&table_ref);
                let Some(name) = model_name_at(&models.borrow(), row) else {
                    set_status(&status, if row < 0 {
                        "Sélectionnez d'abord un modèle à supprimer."
                    } else {
                        "Ligne invalide."
                    });
                    return;
                };
                // Aucun emprunt de `window` n'est conservé ici : le dialogue
                // modal tourne dans sa propre boucle d'événements.
                if !confirm(&format!("Supprimer « {name} » ?")) {
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

    // ---- Barre de progression (pleine largeur, centree en hauteur) ----------
    let bar_row = Widget::new().parent(&central).build();
    bar_row.set_max_size(16_777_215, 58);
    bar_row.set_min_size(0, 46);
    let mut bar_v = VBoxLayout::with_parent(&bar_row);
    bar_v.set_spacing(0);
    bar_v.add_spacer(Spacer::vertical_expanding());
    bar_v.add(wrap_scroll(&bar, wp));
    bar_v.add_spacer(Spacer::vertical_expanding());
    root.add(bar_row);

    // ---- Logs / statut (pleine largeur, sous la barre) ----------------------
    let log_row = Widget::new().parent(&central).build();
    log_row.set_max_size(16_777_215, 130);
    log_row.set_min_size(0, 80);
    let mut log_v = VBoxLayout::with_parent(&log_row);
    log_v.set_spacing(0);
    log_v.add(wrap_scroll(&status, wp));
    root.add(log_row);

    // Enveloppe défilante.
    let central_scroll =
        ScrollArea::new().parent(wp).set_widget_resizable(true).build();
    central_scroll.set_widget(&central);
    w.set_central_widget(&central_scroll);

    let layouts = Layouts {
        root,
        srv: srv_l,
        api: api_l,
        dl: dl_g,
        actions: actions_l,
        modbox: mod_l,
        bar_row: bar_v,
        log_row: log_v,
        central,
    };

    (ui, layouts)
}

// ------------------------------------------------------------------
fn main() {
    let app = Application::new();
    // Icone de fenetre / barre des taches : on pointe un fichier reel.
    // (Le nom de theme seul n'etait pas resolu par le menu -> icone vide.)
    if let Some(p) = app_icon_path() {
        app.set_icon(&p);
    }
    // Indispensable sous Wayland/KDE : associe la fenetre au .desktop pour que
    // le compositeur affiche son icone dans le sélecteur de taches.
    app.set_desktop_file_name("ollama-manager-qt");
    // Clé API enregistrée : appliquée à toutes les requêtes HTTP.
    ollama::set_api_key(Some(load_config().1));
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
            let st = Rc::downgrade(&ui.status);
            refresh_list(&tx, &url, &st);
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
            // Weak pour les widgets : la pompe ne doit pas prolonger leur vie.
            let table = Rc::downgrade(&ui.table);
            let bar = Rc::downgrade(&ui.bar);
            let status = Rc::downgrade(&ui.status);
            let models = Rc::clone(&ui.models); // Vec, pas un widget
            move || {
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        WorkerMsg::List(Some(payload), _) => {
                            let parsed: Vec<Model> =
                                serde_json::from_str(&payload).unwrap_or_default();
                            let count = parsed.len();
                            let (Some(t), Some(b), Some(st)) =
                                (table.upgrade(), bar.upgrade(), status.upgrade())
                            else {
                                continue;
                            };
                            let ui_l = Ui {
                                table: t,
                                bar: b,
                                status: st,
                                models: models.clone(),
                            };
                            fill_table(&ui_l, parsed);
                            set_status(
                                &status,
                                &format!("{} modèle(s) chargé(s).", count),
                            );
                        }
                        WorkerMsg::List(None, err) => {
                            set_status(&status, &format!("Erreur : {err}"));
                        }
                        WorkerMsg::Progress { name, pct, status: st } => {
                            if let Some(b) = bar.upgrade() {
                                b.borrow_mut().set_value(pct as i32);
                            }
                            set_status(&status, &format!("[{name}] {st} — {pct}%"));
                        }
                        WorkerMsg::ActionDone { name, ok, message } => {
                            // PAS de boîte modale ici : ce code tourne dans le
                            // callback du timer Qt, et un exec() imbriqué
                            // ré-entrait la boucle d'événements (le dialogue
                            // pouvait être détruit sous ses propres connexions,
                            // d'où le spam QObject::disconnect et l'action qui
                            // semblait ne pas aboutir). Le statut affiche le
                            // résultat, sans bloquer la pompe.
                            reset_bar(&bar);
                            let mark = if ok { "✔" } else { "✘" };
                            set_status(&status, &format!("{mark} {name} : {message}"));
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

    // Crochet de test (inactif en usage normal) : ferme la fenetre apres N ms
    // pour sortir de app.exec() par le meme chemin qu'un clic sur la croix.
    let _auto_close = std::env::var("OM_AUTO_CLOSE_MS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .map(|ms| {
            let w = Rc::clone(&window);
            Timer::new(ms as i32)
                .on_timeout(move || {
                    w.borrow().close();
                })
                .build()
        });

    app.exec();

    // Les callbacks ne detiennent que des `Weak` : `ui` est le seul
    // proprietaire des widgets, donc ses wrappers sont detruits AVANT que Qt
    // libere les objets C++ (plus de « wildcard call disconnects from
    // destroyed signal » a la fermeture).
}

// ------------------------------------------------------------------
// Tests unitaires (logique de persistance, sans GUI)
// ------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boutons_dialogue_distincts() {
        // Regression : 0x400 n'est PAS "Non" mais "Ok" ; l'ancien code attendait
        // 0x4000 (Oui) et un clic sur "Ok" annulait la suppression en silence.
        assert_eq!(BTN_YES, 0x4000);
        assert_eq!(BTN_CANCEL, 0x0040_0000);
        assert_ne!(BTN_CANCEL, 0x400);
        assert_ne!(BTN_YES, BTN_CANCEL);
        // Les deux boutons doivent figurer dans le masque transmis a Qt.
        let mask = BTN_YES | BTN_CANCEL;
        assert_eq!(mask & BTN_YES, BTN_YES);
        assert_eq!(mask & BTN_CANCEL, BTN_CANCEL);
    }

    #[test]
    fn model_name_at_tests() {
        let mk = |n: &str| Model {
            name: n.to_string(),
            size: 0,
            modified_at: String::new(),
            params: String::new(),
            quant: String::new(),
            caps: String::new(),
        };
        let models = vec![mk("llama3.2:3b"), mk("qwen3:8b")];
        // Sélection valide -> nom exact (c'est ce nom qui part dans DELETE).
        assert_eq!(model_name_at(&models, 0).as_deref(), Some("llama3.2:3b"));
        assert_eq!(model_name_at(&models, 1).as_deref(), Some("qwen3:8b"));
        // Aucune sélection.
        assert_eq!(model_name_at(&models, -1), None);
        // Ligne hors bornes (liste rafraîchie entre-temps).
        assert_eq!(model_name_at(&models, 2), None);
        // Liste vide.
        assert_eq!(model_name_at(&[], 0), None);
    }

    #[test]
    fn ini_api_key_simple() {
        assert_eq!(
            parse_ini_api_key("[General]\nserverUrl=http://x:11434\napiKey=sk-abc123\n"),
            Some("sk-abc123".to_string())
        );
    }

    #[test]
    fn ini_api_key_absente() {
        // Pas de ligne apiKey => None (clé facultative).
        assert_eq!(
            parse_ini_api_key("[General]\nserverUrl=http://x:11434\n"),
            None
        );
    }

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
