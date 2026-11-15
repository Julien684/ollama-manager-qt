//! Ollama Model Manager — GUI Qt6 (via `qtrs`), binaire unique.
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

use qtrs::prelude::*;
use qtrs::{Application, MessageBox, TabWidget, Timer};

mod ollama;
use ollama::{fmt_bytes, fmt_context, fmt_date, fmt_remaining, delete_model, get_version,
             list_models, list_running, load_model, processor_label, pull_model,
             unload_model, Model, RunningModel};

// ------------------------------------------------------------------
// Messages worker → UI
// ------------------------------------------------------------------
enum WorkerMsg {
    /// Résultats d'un `GET /api/tags` : (série JSON des modèles, message d'erreur si échec).
    List(Option<String>, String),
    /// Résultats d'un `GET /api/ps` : (série JSON des modèles chargés, erreur).
    Running(Option<String>, String),
    /// Progression d'un pull : (nom, pct, statut).
    Progress { name: String, pct: u32, status: String },
    /// Fin d'une action (pull / delete) : (nom, ok, message).
    ActionDone { name: String, ok: bool, message: String },
    /// Fin d'un chargement / déchargement depuis l'onglet Moniteur.
    MonitorDone { ok: bool, message: String },
}

/// Un seul `GET /api/ps` à la fois : sans ce verrou, un serveur injoignable
/// (timeout 120 s) ferait s'empiler un thread toutes les 5 s.
static PS_BUSY: AtomicBool = AtomicBool::new(false);


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

/// Parse la clé `keepAlive` (durée de maintien en mémoire) d'un INI Qt.
pub fn parse_ini_keep_alive(content: &str) -> Option<String> {
    parse_ini_key(content, "keepAlive")
}

/// Lit (URL du serveur, clé API, keep_alive) depuis la configuration.
fn load_config() -> (String, String, String) {
    let content = std::fs::read_to_string(config_path()).unwrap_or_default();
    (
        parse_ini_server_url(&content).unwrap_or_default(),
        parse_ini_api_key(&content).unwrap_or_default(),
        parse_ini_keep_alive(&content).unwrap_or_default(),
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

/// Écrit l'URL, la clé API et le keep_alive (une clé vide retire `apiKey`).
fn save_config(url: &str, api_key: &str, keep_alive: &str) {
    let path = config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut out = String::from("[General]\n");
    out.push_str(&format!("serverUrl={}\n", ini_value(url)));
    if !api_key.trim().is_empty() {
        out.push_str(&format!("apiKey={}\n", ini_value(api_key.trim())));
    }
    if !keep_alive.trim().is_empty() {
        out.push_str(&format!("keepAlive={}\n", ini_value(keep_alive.trim())));
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
        .window_title("Ollama Model Manager")
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
    /// Liste complète des modèles (source de vérité, avant filtrage).
    models: Rc<RefCell<Vec<Model>>>,
    /// Vue filtrée affichée dans l'onglet Gestion (index de ligne = position).
    g_view: Rc<RefCell<Vec<Model>>>,
    /// Vue filtrée affichée dans l'onglet Moniteur.
    m_view: Rc<RefCell<Vec<Model>>>,
    /// Champs de filtre : pointeurs C++ pour lire le texte en direct.
    g_filter: *mut qtrs::ffi::QLineEdit,
    m_filter: *mut qtrs::ffi::QLineEdit,
    /// Dernières valeurs lues (on ne refiltre qu'au changement).
    g_filter_last: Rc<RefCell<String>>,
    m_filter_last: Rc<RefCell<String>>,
    /// Chargement en cours : (nom, taille, instant de départ) → compteur.
    loading: Rc<RefCell<Option<(String, u64, std::time::Instant)>>>,
    /// Barre du Moniteur (mode indéterminé pendant un chargement).
    m_bar: Rc<RefCell<ProgressBar>>,
    /// Onglet Moniteur : modèles chargés (`ollama ps`).
    run_table: Rc<RefCell<TableWidget>>,
    /// Onglet Moniteur : modèles disponibles au chargement.
    avail_table: Rc<RefCell<TableWidget>>,
    /// Statut propre à l'onglet Moniteur (ne pas polluer celui de Gestion).
    m_status: Rc<RefCell<Label>>,
    /// Modèles actuellement chargés (index de ligne = position dans ce Vec).
    running: Rc<RefCell<Vec<RunningModel>>>,
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
    /// Layouts des deux pages d'onglets et des boîtes du Moniteur.
    #[allow(dead_code)]
    g_page: VBoxLayout,
    #[allow(dead_code)]
    m_page: VBoxLayout,
    #[allow(dead_code)]
    ka_grid: GridLayout,
    #[allow(dead_code)]
    run_box: VBoxLayout,
    #[allow(dead_code)]
    run_actions: HBoxLayout,
    #[allow(dead_code)]
    avail_box: VBoxLayout,
    #[allow(dead_code)]
    avail_actions: HBoxLayout,
    /// Widget central sans parent : SANS lui, qtrs supprime le C++ du widget
    /// (donc toute la sous-arborescence) à la fin de `build_ui` → segfault dans
    /// `app.exec()`. Doit vivre aussi longtemps que la fenêtre.
    #[allow(dead_code)]
    central: Widget,
}

// ------------------------------------------------------------------
// Actions (utilisées par les boutons ET par le timer)
// ------------------------------------------------------------------

/// Vrai si le nom contient le filtre (insensible à la casse). Un filtre vide
/// accepte tout.
pub fn matches_filter(name: &str, needle: &str) -> bool {
    let n = needle.trim().to_lowercase();
    n.is_empty() || name.to_lowercase().contains(&n)
}

/// Applique le filtre à une liste de modèles.
pub fn filtered(models: &[Model], needle: &str) -> Vec<Model> {
    models
        .iter()
        .filter(|m| matches_filter(&m.name, needle))
        .cloned()
        .collect()
}

/// Remplit la table de l'onglet Gestion avec la liste (déjà filtrée) reçue.
fn fill_table(table: &TableWidget, view: &Rc<RefCell<Vec<Model>>>, list: Vec<Model>) {
    view.borrow_mut().clone_from(&list);
    table.set_row_count(list.len() as i32);
    for (i, m) in list.iter().enumerate() {
        table.set_item(i as i32, 0, &m.name);
        table.set_item(i as i32, 1, &fmt_bytes(m.size));
        let meta = format!("{} {}", m.params, m.quant).trim().to_string();
        table.set_item(i as i32, 2, &meta);
        table.set_item(i as i32, 3, &fmt_date(&m.modified_at));
        table.set_item(i as i32, 4, &m.caps);
    }
}

/// Remplit la liste « modèles disponibles » de l'onglet Moniteur.
fn fill_avail(table: &TableWidget, view: &Rc<RefCell<Vec<Model>>>, list: Vec<Model>) {
    view.borrow_mut().clone_from(&list);
    table.set_row_count(list.len() as i32);
    for (i, m) in list.iter().enumerate() {
        table.set_item(i as i32, 0, &m.name);
        table.set_item(i as i32, 1, &fmt_bytes(m.size));
        let meta = format!("{} {}", m.params, m.quant).trim().to_string();
        table.set_item(i as i32, 2, &meta);
        table.set_item(i as i32, 3, &m.caps);
    }
}

/// Remplit l'état « modèles chargés » — colonnes de `ollama ps` (nom, taille,
/// processeur, expiration) plus la taille du contexte allouée.
fn fill_running(table: &TableWidget, list: &[RunningModel], now: i64) {
    table.set_row_count(list.len() as i32);
    for (i, m) in list.iter().enumerate() {
        table.set_item(i as i32, 0, &m.name);
        table.set_item(i as i32, 1, &fmt_bytes(m.size));
        table.set_item(i as i32, 2, &processor_label(m.size, m.size_vram));
        table.set_item(i as i32, 3, &fmt_context(m.context_length));
        table.set_item(i as i32, 4, &fmt_remaining(&m.expires_at, now));
    }
}

/// Nom du modèle chargé à la ligne `row` (-1 = aucune sélection).
pub fn running_name_at(models: &[RunningModel], row: i32) -> Option<String> {
    if row < 0 {
        return None;
    }
    models.get(row as usize).map(|m| m.name.clone())
}

/// Secondes Unix courantes (pour « expire dans … »).
fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Lance une lecture de `GET /api/ps` dans un thread.
fn refresh_running(tx: &mpsc::Sender<WorkerMsg>, url: &str) {
    let tx = tx.clone();
    let url = url.to_string();
    std::thread::spawn(move || {
        let (payload, err) = match list_running(&url) {
            Ok(v) => (
                Some(serde_json::to_string(&v).unwrap_or_default()),
                String::new(),
            ),
            Err(e) => (None, e.to_string()),
        };
        let _ = tx.send(WorkerMsg::Running(payload, err));
    });
}

/// Lance une lecture de `GET /api/tags` dans un thread (rafraîchissement).
fn spawn_list(tx: mpsc::Sender<WorkerMsg>, url: String) {
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
    // Onglet Moniteur : état courant (`ollama ps`) + modèles disponibles.
    let run_table = Rc::new(RefCell::new(TableWidget::new(0, 5).build()));
    let avail_table = Rc::new(RefCell::new(TableWidget::new(0, 4).build()));
    let m_status = Rc::new(RefCell::new(Label::new("Aucun modèle chargé.").build()));
    let running = Rc::new(RefCell::new(Vec::new()));
    // Filtres : vues affichées + état du chargement en cours.
    let g_view = Rc::new(RefCell::new(Vec::new()));
    let m_view = Rc::new(RefCell::new(Vec::new()));
    let g_filter_last = Rc::new(RefCell::new(String::new()));
    let m_filter_last = Rc::new(RefCell::new(String::new()));
    let loading: Rc<RefCell<Option<(String, u64, std::time::Instant)>>> =
        Rc::new(RefCell::new(None));
    let m_bar = Rc::new(RefCell::new(
        ProgressBar::new().range(0, 100).value(0).format("%p %").build(),
    ));
    m_bar.borrow_mut().set_has_parent();

    table.borrow_mut().set_has_parent();
    bar.borrow_mut().set_has_parent();
    status.borrow_mut().set_has_parent();
    run_table.borrow_mut().set_has_parent();
    avail_table.borrow_mut().set_has_parent();
    m_status.borrow_mut().set_has_parent();
    table.borrow_mut().set_horizontal_header_labels(&[
        "Modèle", "Taille", "Paramètres", "Modifié", "Capacités",
    ]);
    table.borrow_mut().set_column_width(0, 380);
    table.borrow_mut().set_column_width(1, 80);
    table.borrow_mut().set_column_width(2, 120);
    table.borrow_mut().set_column_width(3, 90);
    table.borrow_mut().set_column_width(4, 260);
    // Mêmes colonnes que `ollama ps` (NAME / SIZE / PROCESSOR / UNTIL).
    run_table.borrow_mut().set_horizontal_header_labels(&[
        "Modèle", "Taille", "Processeur", "Contexte", "Expire dans",
    ]);
    run_table.borrow_mut().set_column_width(0, 400);
    run_table.borrow_mut().set_column_width(1, 90);
    run_table.borrow_mut().set_column_width(2, 150);
    run_table.borrow_mut().set_column_width(3, 90);
    run_table.borrow_mut().set_column_width(4, 120);
    avail_table.borrow_mut().set_horizontal_header_labels(&[
        "Modèle", "Taille", "Paramètres", "Capacités",
    ]);
    avail_table.borrow_mut().set_column_width(0, 380);
    avail_table.borrow_mut().set_column_width(1, 90);
    avail_table.borrow_mut().set_column_width(2, 130);
    avail_table.borrow_mut().set_column_width(3, 220);

    let w = window.borrow();
    let wp: &dyn AsWidget = &*w;
    let mut ui = Ui {
        table: table.clone(),
        bar: bar.clone(),
        status: status.clone(),
        models: models.clone(),
        g_view: g_view.clone(),
        m_view: m_view.clone(),
        g_filter: std::ptr::null_mut(),
        m_filter: std::ptr::null_mut(),
        g_filter_last: g_filter_last.clone(),
        m_filter_last: m_filter_last.clone(),
        loading: loading.clone(),
        m_bar: m_bar.clone(),
        run_table: run_table.clone(),
        avail_table: avail_table.clone(),
        m_status: m_status.clone(),
        running: running.clone(),
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

    // ---- Widget central + onglets ------------------------------------------
    let mut central = Widget::new().build();
    // Sans ce drapeau, le wrapper Rust detruit le C++ a la fin de `build_ui`
    // alors que QMainWindow garde le pointeur -> segfault au premier paint.
    central.set_has_parent();
    let mut root = VBoxLayout::with_parent(&central);
    root.set_contents_margins(12, 12, 12, 12);
    root.set_spacing(10);

    // Onglets : « Gestion » (interface historique) et « Moniteur » (nouveau).
    let mut tabs = TabWidget::new()
        .on_current_changed({
            let tx = tx.clone();
            move |idx| {
                // Afficher l'onglet Moniteur rafraichit l'etat (`ollama ps`).
                if idx == 1 {
                    let url = load_server_url();
                    if !url.trim().is_empty() {
                        refresh_running(&tx, &url);
                    }
                }
            }
        })
        .build();

    // Page « Gestion » : tout l'ecran precedent, inchange.
    let g_page = Widget::new().build();
    let mut g_l = VBoxLayout::with_parent(&g_page);
    g_l.set_spacing(10);

    // ---- Clé API facultative (créée tôt pour être lue par « Tester ») ------
    let api_title = Label::new(
        "Clé API (facultative — laisser vide si le serveur n'en exige pas)",
    )
    .parent(&g_page)
    .build();
    let api_row = Widget::new().parent(&g_page).build();
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
            .parent(&g_page)
            .build();
    g_l.add(srv_title);
    let srv_row = Widget::new().parent(&g_page).build();
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
                save_config(&url, &key, &load_config().2);
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

    g_l.add(srv_row);
    g_l.add(api_title);
    g_l.add(api_row);

    // ---- Télécharger un modèle ----------------------------------------------
    let dl_box =
        GroupBox::new("Télécharger un modèle").parent(&g_page).build();
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
    g_l.add(dl_box);

    // ---- Modèles locaux ------------------------------------------------------
    let mod_box =
        GroupBox::new("Modèles locaux").parent(&g_page).build();
    let mut mod_l = VBoxLayout::with_parent(&mod_box);
    mod_l.set_spacing(8);

    // Filtre en direct (appliqué par la pompe, LineEdit n'ayant pas de signal).
    // Champ créé VIDE : `LineEdit::new("…")` écrit du texte réel, pas un
    // placeholder — un libellé séparé évite que le filtre ne s'applique à lui-même.
    let g_filter_lbl = Label::new("Filtrer les modèles (nom ou tag)")
        .parent(&mod_box)
        .build();
    mod_l.add(g_filter_lbl);
    let g_filter = LineEdit::new("").parent(&mod_box).build();
    let g_filter_ptr = g_filter.widget_ptr() as *mut qtrs::ffi::QLineEdit;
    mod_l.add(g_filter);

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
            // Vue filtrée : l'index de ligne correspond à ce qui est affiché.
            let models = g_view.clone();
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
            let models = g_view.clone();
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
    g_l.add(mod_box);

    // ---- Barre de progression (ligne compacte, sans espace vertical) -------
    // Pas de Widget intermediaire ni de spacers verticaux : la barre occupe
    // une seule ligne de 30 px, comme dans l'onglet Moniteur.
    let bar_sa = wrap_scroll(&bar, wp);
    unsafe { qtrs::ffi::QWidget_setFixedHeight(bar_sa.widget_ptr(), 30) };
    g_l.add(bar_sa);

    // ---- Statut (une seule ligne, sous la barre) ---------------------------
    let status_sa = wrap_scroll(&status, wp);
    unsafe { qtrs::ffi::QWidget_setFixedHeight(status_sa.widget_ptr(), 30) };
    g_l.add(status_sa);

    // ================= Page « Moniteur » ====================================
    let m_page = Widget::new().build();
    let mut m_l = VBoxLayout::with_parent(&m_page);
    m_l.set_spacing(10);

    // ---- keep_alive ---------------------------------------------------------
    let ka_box = GroupBox::new("Maintien en mémoire (keep_alive)")
        .parent(&m_page)
        .build();
    let mut ka_g = GridLayout::with_parent(&ka_box);
    let ka_title = Label::new(
        "Durée appliquée au chargement — ex. 5m, 30m, 2h ; -1 = ne jamais décharger",
    )
    .parent(&ka_box)
    .build();
    ka_g.add(ka_title, 0, 0, 1, 2);
    let saved_ka = load_config().2;
    let ka_default = if saved_ka.trim().is_empty() {
        "5m".to_string()
    } else {
        saved_ka
    };
    let ka_edit = LineEdit::new(&ka_default).parent(&ka_box).build();
    let ka_ptr = ka_edit.widget_ptr() as *mut qtrs::ffi::QLineEdit;
    ka_g.add(ka_edit, 1, 0, 1, 1);
    let ka_hint = Label::new("Appliqué au prochain chargement")
        .parent(&ka_box)
        .build();
    ka_g.add(ka_hint, 1, 1, 1, 1);
    m_l.add(ka_box);

    // ---- Modèles chargés — état actuel (`ollama ps`) ------------------------
    let run_box = GroupBox::new("Modèles chargés — état actuel (ollama ps)")
        .parent(&m_page)
        .build();
    let mut run_l = VBoxLayout::with_parent(&run_box);
    run_l.set_spacing(8);
    let run_actions_row = Widget::new().parent(&run_box).build();
    let mut run_a = HBoxLayout::with_parent(&run_actions_row);
    run_a.set_spacing(8);

    let run_refresh_btn = PushButton::new("Actualiser")
        .on_clicked({
            let tx = tx.clone();
            let mstatus = Rc::downgrade(&m_status);
            move || {
                let url = load_server_url();
                if url.trim().is_empty() {
                    set_status(&mstatus, "Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                set_status(&mstatus, "Lecture de l'état…");
                refresh_running(&tx, &url);
            }
        })
        .parent(&run_actions_row)
        .build();
    run_a.add(run_refresh_btn);

    let unload_btn = PushButton::new("Décharger")
        .on_clicked({
            let tx = tx.clone();
            let mstatus = Rc::downgrade(&m_status);
            let running = running.clone();
            let run_ref = Rc::downgrade(&run_table);
            move || {
                let row = table_row(&run_ref);
                let Some(name) = running_name_at(&running.borrow(), row) else {
                    set_status(&mstatus, if row < 0 {
                        "Sélectionnez un modèle chargé."
                    } else {
                        "Ligne invalide."
                    });
                    return;
                };
                let url = load_server_url();
                if url.trim().is_empty() {
                    set_status(&mstatus, "Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                set_status(&mstatus, &format!("Déchargement de {name}…"));
                let tw = tx.clone();
                std::thread::spawn(move || {
                    let (ok, message) = match unload_model(&url, &name) {
                        Ok(()) => (true, format!("{name} déchargé")),
                        Err(e) => (false, e.to_string()),
                    };
                    let _ = tw.send(WorkerMsg::MonitorDone { ok, message });
                });
            }
        })
        .parent(&run_actions_row)
        .build();
    run_a.add(unload_btn);

    run_l.add(run_actions_row);
    // Barre de chargement : mode indéterminé pendant le chargement d'un modèle
    // (Ollama n'expose aucune progression), puis 100 % à la fin.
    let m_bar_sa = wrap_scroll(&m_bar, wp);
    unsafe { qtrs::ffi::QWidget_setFixedHeight(m_bar_sa.widget_ptr(), 30) };
    run_l.add(m_bar_sa);
    run_l.add(wrap_scroll(&run_table, wp));
    m_l.add(run_box);

    // ---- Modèles disponibles — à charger ------------------------------------
    let avail_box = GroupBox::new("Modèles disponibles — à charger")
        .parent(&m_page)
        .build();
    let mut avail_l = VBoxLayout::with_parent(&avail_box);
    avail_l.set_spacing(8);

    // Filtre en direct de la liste des modèles disponibles (champ vide, voir
    // la note de l'onglet Gestion : `LineEdit::new` écrit du texte réel).
    let m_filter_lbl = Label::new("Filtrer les modèles (nom ou tag)")
        .parent(&avail_box)
        .build();
    avail_l.add(m_filter_lbl);
    let m_filter = LineEdit::new("").parent(&avail_box).build();
    let m_filter_ptr = m_filter.widget_ptr() as *mut qtrs::ffi::QLineEdit;
    avail_l.add(m_filter);
    let avail_actions_row = Widget::new().parent(&avail_box).build();
    let mut avail_a = HBoxLayout::with_parent(&avail_actions_row);
    avail_a.set_spacing(8);

    let load_btn = PushButton::new("Charger")
        .on_clicked({
            let tx = tx.clone();
            let mstatus = Rc::downgrade(&m_status);
            let models = m_view.clone();
            let avail_ref = Rc::downgrade(&avail_table);
            let api_p = api_ptr;
            let loading = loading.clone();
            let m_bar = m_bar.clone();
            move || {
                let row = table_row(&avail_ref);
                let Some(name) = model_name_at(&models.borrow(), row) else {
                    set_status(&mstatus, if row < 0 {
                        "Sélectionnez un modèle à charger."
                    } else {
                        "Ligne invalide."
                    });
                    return;
                };
                let size = models.borrow().get(row as usize).map(|m| m.size).unwrap_or(0);
                let url = load_server_url();
                if url.trim().is_empty() {
                    set_status(&mstatus, "Enregistrez d'abord l'adresse du serveur.");
                    return;
                }
                let ka = read_line_edit(ka_ptr);
                let ka_txt = if ka.trim().is_empty() {
                    "5m".to_string()
                } else {
                    ka.trim().to_string()
                };
                // Mémorise la durée choisie pour les prochaines ouvertures.
                save_config(&url, &read_line_edit(api_p), &ka_txt);
                // Barre indéterminée + compteur : Ollama n'expose aucune
                // progression pendant le chargement, mais l'utilisateur voit
                // que ça avance (temps écoulé) au lieu de croire à un blocage.
                m_bar.borrow().set_range(0, 0);
                *loading.borrow_mut() = Some((name.clone(), size, std::time::Instant::now()));
                set_status(
                    &mstatus,
                    &format!("Chargement de {name} ({}) — 0 s", fmt_bytes(size)),
                );
                let tw = tx.clone();
                std::thread::spawn(move || {
                    let (ok, message) = match load_model(&url, &name, &ka_txt) {
                        Ok(()) => (true, format!("{name} chargé (keep_alive {ka_txt})")),
                        Err(e) => (false, e.to_string()),
                    };
                    let _ = tw.send(WorkerMsg::MonitorDone { ok, message });
                });
            }
        })
        .parent(&avail_actions_row)
        .build();
    avail_a.add(load_btn);

    avail_l.add(avail_actions_row);
    avail_l.add(wrap_scroll(&avail_table, wp));
    m_l.add(avail_box);

    // Statut propre à l'onglet Moniteur (résultat des actions).
    let m_status_sa = wrap_scroll(&m_status, wp);
    unsafe { qtrs::ffi::QWidget_setFixedHeight(m_status_sa.widget_ptr(), 34) };
    m_l.add(m_status_sa);

    // ---- Assemblage des onglets --------------------------------------------
    tabs.add_tab(Box::new(g_page), "Gestion");
    tabs.add_tab(Box::new(m_page), "Moniteur");
    root.add(tabs);
    w.set_central_widget(&central);

    // Pointeurs des champs de filtre (lus par la pompe).
    ui.g_filter = g_filter_ptr;
    ui.m_filter = m_filter_ptr;

    let layouts = Layouts {
        root,
        srv: srv_l,
        api: api_l,
        dl: dl_g,
        actions: actions_l,
        modbox: mod_l,
        g_page: g_l,
        m_page: m_l,
        ka_grid: ka_g,
        run_box: run_l,
        run_actions: run_a,
        avail_box: avail_l,
        avail_actions: avail_a,
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
            .window_title("Ollama Model Manager")
            .size(1060, 720)
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
            refresh_running(&tx, &url);
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
            // Filtres (texte lu en direct) et état de chargement.
            let g_view = Rc::clone(&ui.g_view);
            let m_view = Rc::clone(&ui.m_view);
            let g_filter = ui.g_filter;
            let m_filter = ui.m_filter;
            let g_filter_last = Rc::clone(&ui.g_filter_last);
            let m_filter_last = Rc::clone(&ui.m_filter_last);
            let loading = Rc::clone(&ui.loading);
            let m_bar = Rc::downgrade(&ui.m_bar);
            // Onglet Moniteur.
            let run_table = Rc::downgrade(&ui.run_table);
            let avail_table = Rc::downgrade(&ui.avail_table);
            let m_status = Rc::downgrade(&ui.m_status);
            let running = Rc::clone(&ui.running);
            move || {
                while let Ok(msg) = rx.try_recv() {
                    match msg {
                        WorkerMsg::List(Some(payload), _) => {
                            let parsed: Vec<Model> =
                                serde_json::from_str(&payload).unwrap_or_default();
                            let count = parsed.len();
                            let (Some(t), Some(_st)) = (table.upgrade(), status.upgrade())
                            else {
                                continue;
                            };
                            models.borrow_mut().clone_from(&parsed);
                            // Filtres courants (relus ici : la liste peut
                            // arriver après une saisie dans le champ).
                            let ng = if g_filter.is_null() {
                                String::new()
                            } else {
                                read_line_edit(g_filter)
                            };
                            let nm = if m_filter.is_null() {
                                String::new()
                            } else {
                                read_line_edit(m_filter)
                            };
                            *g_filter_last.borrow_mut() = ng.clone();
                            *m_filter_last.borrow_mut() = nm.clone();
                            fill_table(&t.borrow(), &g_view, filtered(&parsed, &ng));
                            // La liste alimente aussi l'onglet Moniteur.
                            if let Some(at) = avail_table.upgrade() {
                                fill_avail(&at.borrow(), &m_view, filtered(&parsed, &nm));
                            }
                            set_status(
                                &status,
                                &format!("{} modèle(s) chargé(s).", count),
                            );
                        }
                        WorkerMsg::List(None, err) => {
                            set_status(&status, &format!("Erreur : {err}"));
                        }
                        WorkerMsg::Running(Some(payload), _) => {
                            let parsed: Vec<RunningModel> =
                                serde_json::from_str(&payload).unwrap_or_default();
                            let count = parsed.len();
                            if let Some(rt) = run_table.upgrade() {
                                fill_running(&rt.borrow(), &parsed, now_epoch());
                            }
                            running.borrow_mut().clone_from(&parsed);
                            set_status(
                                &m_status,
                                &if count == 0 {
                                    "Aucun modèle chargé.".to_string()
                                } else {
                                    format!("{count} modèle(s) chargé(s).")
                                },
                            );
                        }
                        WorkerMsg::Running(None, err) => {
                            set_status(&m_status, &format!("Erreur : {err}"));
                        }
                        WorkerMsg::MonitorDone { ok, message } => {
                            let mark = if ok { "✔" } else { "✘" };
                            // Fin du chargement : barre à 100 % (ou 0 en échec).
                            *loading.borrow_mut() = None;
                            if let Some(b) = m_bar.upgrade() {
                                b.borrow().set_range(0, 100);
                                b.borrow().set_value(if ok { 100 } else { 0 });
                            }
                            set_status(&m_status, &format!("{mark} {message}"));
                            let url = load_server_url();
                            if !url.trim().is_empty() {
                                refresh_running(&tx2, &url);
                            }
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
                                spawn_list(tx2.clone(), url.clone());
                                refresh_running(&tx2, &url);
                            }
                        }
                    }
                }

                // ---- Filtres : refiltrer seulement quand le texte change ----
                if !g_filter.is_null() {
                    let needle = read_line_edit(g_filter);
                    if *g_filter_last.borrow() != needle {
                        *g_filter_last.borrow_mut() = needle.clone();
                        if let Some(t) = table.upgrade() {
                            let list = filtered(&models.borrow(), &needle);
                            fill_table(&t.borrow(), &g_view, list);
                        }
                    }
                }
                if !m_filter.is_null() {
                    let needle = read_line_edit(m_filter);
                    if *m_filter_last.borrow() != needle {
                        *m_filter_last.borrow_mut() = needle.clone();
                        if let Some(at) = avail_table.upgrade() {
                            let list = filtered(&models.borrow(), &needle);
                            fill_avail(&at.borrow(), &m_view, list);
                        }
                    }
                }

                // ---- Compteur de chargement (Ollama ne donne pas de %) ------
                if let Some((name, size, t0)) = loading.borrow().as_ref() {
                    let secs = t0.elapsed().as_secs();
                    set_status(
                        &m_status,
                        &format!(
                            "⏳ Chargement de {name} ({}) — {secs} s",
                            fmt_bytes(*size)
                        ),
                    );
                }
            }
        })
        .build();

    // ---- Timer de surveillance : rafraîchit `ollama ps` toutes les 5 s ------
    // Déclaré à portée de `main` (comme la pompe) pour vivre jusqu'à la fin
    // de `app.exec()`.
    let _ps_timer = Timer::new(5000)
        .on_timeout({
            let tx = tx.clone();
            move || {
                let url = load_server_url();
                if url.trim().is_empty() {
                    return;
                }
                // Un seul appel en vol : un serveur injoignable bloque 120 s.
                if PS_BUSY.swap(true, Ordering::SeqCst) {
                    return;
                }
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let (payload, err) = match list_running(&url) {
                        Ok(v) => (
                            Some(serde_json::to_string(&v).unwrap_or_default()),
                            String::new(),
                        ),
                        Err(e) => (None, e.to_string()),
                    };
                    let _ = tx.send(WorkerMsg::Running(payload, err));
                    PS_BUSY.store(false, Ordering::SeqCst);
                });
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
    fn filtre_insensible_a_la_casse() {
        let mk = |n: &str| Model {
            name: n.to_string(),
            size: 0,
            modified_at: String::new(),
            params: String::new(),
            quant: String::new(),
            caps: String::new(),
        };
        let models = vec![
            mk("llama3.2:3b"),
            mk("Qwen2.5:7b"),
            mk("nomic-embed-text:latest"),
        ];
        // Filtre vide ou blanc => tout passe.
        assert_eq!(filtered(&models, "").len(), 3);
        assert_eq!(filtered(&models, "   ").len(), 3);
        // Insensible à la casse.
        assert_eq!(filtered(&models, "QWEN").len(), 1);
        assert_eq!(filtered(&models, "qwen")[0].name, "Qwen2.5:7b");
        // Sous-chaîne, tag compris.
        assert_eq!(filtered(&models, "embed").len(), 1);
        assert_eq!(filtered(&models, "3b").len(), 1);
        // Aucun résultat.
        assert_eq!(filtered(&models, "absent").len(), 0);
        // La correspondance porte sur le nom, pas sur les métadonnées.
        assert!(!matches_filter("llama3.2:3b", "Q4"));
    }

    #[test]
    fn running_name_at_tests() {
        let mk = |n: &str| RunningModel {
            name: n.to_string(),
            size: 1,
            size_vram: 1,
            expires_at: String::new(),
            context_length: 0,
        };
        let list = vec![mk("llama3.2:latest")];
        // Sélection valide -> nom exact (c'est lui qui part dans /api/generate).
        assert_eq!(
            running_name_at(&list, 0).as_deref(),
            Some("llama3.2:latest")
        );
        // Aucune sélection / hors bornes / liste vide.
        assert_eq!(running_name_at(&list, -1), None);
        assert_eq!(running_name_at(&list, 1), None);
        assert_eq!(running_name_at(&[], 0), None);
    }

    #[test]
    fn ini_keep_alive_lu() {
        assert_eq!(
            parse_ini_keep_alive("[General]\nserverUrl=http://x:11434\nkeepAlive=30m\n"),
            Some("30m".to_string())
        );
        // Absent => None (l'appelant retombe sur 5m).
        assert_eq!(
            parse_ini_keep_alive("[General]\nserverUrl=http://x:11434\n"),
            None
        );
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
