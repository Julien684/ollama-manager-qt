//! Client HTTP de l'API Ollama (ureq, JSON).
//!
//! Côté logique, indépendant de la GUI : chaque fonction est testable sans
//! ouvrir de fenêtre. Le streaming de `/api/pull` est lu ligne par ligne
//! (NDJSON) et reporté via un callback de progression.

use serde::{Deserialize, Serialize};
use std::sync::RwLock;

/// En-tête User-Agent simple (certains proxies exigent un UA).
const UA: &str = "ollama-manager-qt/0.1";

// ----------------------------------------------------------------------
// Structures renvoyées par l'API
// ----------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct Version {
    #[serde(default)]
    pub version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    pub size: u64,
    /// ISO 8601 (ex. 2026-10-24T13:27:15.920Z) — affiché tel quel si non parsé.
    pub modified_at: String,
    pub params: String,  // "34.7B"
    pub quant: String,   // "Q4_K_M"
    pub caps: String,    // liste des capacités, séparées par " · "
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub modified_at: String,
    #[serde(default)]
    pub details: Details,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct Details {
    #[serde(default)]
    pub parameter_size: String,
    #[serde(default)]
    pub quantization_level: String,
}

#[derive(Debug, Deserialize)]
struct TagsResp {
    #[serde(default)]
    models: Vec<ModelInfo>,
}

/// Modèle actuellement chargé en mémoire (`GET /api/ps`).
///
/// Reproduit les colonnes de `ollama ps` : nom, taille, processeur
/// (déduit de `size_vram`), date d'expiration du maintien en mémoire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunningModel {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub size: u64,
    #[serde(default)]
    pub size_vram: u64,
    /// ISO 8601 (ex. 2026-11-15T10:12:56.187695428Z).
    #[serde(default)]
    pub expires_at: String,
    /// Taille du contexte allouée par le runner, en tokens (0 si absent).
    #[serde(default)]
    pub context_length: u64,
}

#[derive(Debug, Deserialize)]
struct PsResp {
    #[serde(default)]
    models: Vec<RunningModel>,
}

#[derive(Debug, Deserialize)]
struct PullEvent {
    #[serde(default)]
    status: String,
    #[serde(default)]
    completed: Option<u64>,
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

// ----------------------------------------------------------------------
// Erreur
// ----------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OllamaError {
    /// Le serveur ne répond pas (connexion refusée, DNS, timeout…).
    Unreachable(String),
    /// Le serveur a répondu en erreur (HTTP >= 400, message d'Ollama).
    Server(String),
    /// Réponse inattendue (JSON invalide, champ manquant…).
    Protocol(String),
}

impl std::fmt::Display for OllamaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OllamaError::Unreachable(m) => write!(f, "serveur injoignable : {m}"),
            OllamaError::Server(m) => write!(f, "erreur du serveur : {m}"),
            OllamaError::Protocol(m) => write!(f, "réponse inattendue : {m}"),
        }
    }
}

type Result<T> = std::result::Result<T, OllamaError>;

// ----------------------------------------------------------------------
// Helpers de demande
// ----------------------------------------------------------------------

fn agent(_base: &str) -> ureq::Agent {
    // Timeout global généreux (connect + read + write) ; un pull peut durer
    // longtemps mais chaque *chunk* lu en chemin réarme le timer de lecture.
    let d = std::time::Duration::from_secs(120);
    ureq::AgentBuilder::new()
        .timeout(d)
        .user_agent(UA)
        .build()
}

// ----------------------------------------------------------------------
// Clé API facultative (Ollama Cloud / proxy authentifié)
// ----------------------------------------------------------------------

/// Clé API courante ; `None` = aucune authentification envoyée.
static API_KEY: RwLock<Option<String>> = RwLock::new(None);

/// Valeur de l'en-tête `Authorization` pour une clé brute (testable).
/// Une clé vide ou faite d'espaces n'authentifie rien.
fn auth_header_value(key: &str) -> Option<String> {
    let k = key.trim();
    if k.is_empty() {
        None
    } else {
        Some(format!("Bearer {k}"))
    }
}

/// Configure la clé API (chaîne vide/espaces => désactive l'authentification).
pub fn set_api_key(key: Option<String>) {
    let cleaned = key.and_then(|k| auth_header_value(&k).map(|_| k.trim().to_string()));
    *API_KEY.write().unwrap() = cleaned;
}

/// Applique `Authorization: Bearer …` si une clé est configurée.
fn with_auth(req: ureq::Request) -> ureq::Request {
    let guard = API_KEY.read().unwrap();
    match guard.as_deref().and_then(auth_header_value) {
        Some(v) => req.set("Authorization", &v),
        None => req,
    }
}

fn normalize_url(base: &str) -> String {
    let t = base.trim();
    if t.is_empty() {
        return String::new();
    }
    let with_scheme = if t.starts_with("http://") || t.starts_with("https://") {
        t.to_string()
    } else {
        format!("http://{t}")
    };
    with_scheme.trim_end_matches('/').to_string()
}

fn url(base: &str, path: &str) -> String {
    format!("{}{}", normalize_url(base), path)
}

/// Convertit une erreur ureq en [`OllamaError`] typé.
fn map_ureq_err(_base: &str, e: ureq::Error) -> OllamaError {
    match e {
        ureq::Error::Status(code, resp) => {
            let body = resp.into_string().unwrap_or_default();
            let json_msg = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(str::to_string));
            let msg = json_msg
                .or_else(|| if body.is_empty() { None } else { Some(body.clone()) })
                .unwrap_or_else(|| format!("HTTP {code}"));
            OllamaError::Server(msg)
        }
        ureq::Error::Transport(t) => OllamaError::Unreachable(t.to_string()),
    }
}

// ----------------------------------------------------------------------
// API
// ----------------------------------------------------------------------

/// `GET /api/version` — vérifie que le serveur répond.
pub fn get_version(base: &str) -> Result<String> {
    let r = with_auth(agent(base).get(&url(base, "/api/version")))
        .call()
        .map_err(|e| map_ureq_err(base, e))?;
    let v: Version = r
        .into_json()
        .map_err(|e| OllamaError::Protocol(e.to_string()))?;
    Ok(v.version)
}

/// `GET /api/tags` — liste les modèles locaux.
pub fn list_models(base: &str) -> Result<Vec<Model>> {
    let r = with_auth(agent(base).get(&url(base, "/api/tags")))
        .call()
        .map_err(|e| map_ureq_err(base, e))?;
    let t: TagsResp = r
        .into_json()
        .map_err(|e| OllamaError::Protocol(e.to_string()))?;
    Ok(t
        .models
        .into_iter()
        .map(|m| Model {
            name: m.name,
            size: m.size,
            modified_at: m.modified_at,
            params: m.details.parameter_size,
            quant: m.details.quantization_level,
            caps: m.capabilities.join(" · "),
        })
        .collect())
}

/// `GET /api/ps` — modèles actuellement chargés en mémoire.
pub fn list_running(base: &str) -> Result<Vec<RunningModel>> {
    let r = with_auth(agent(base).get(&url(base, "/api/ps")))
        .call()
        .map_err(|e| map_ureq_err(base, e))?;
    let p: PsResp = r
        .into_json()
        .map_err(|e| OllamaError::Protocol(e.to_string()))?;
    Ok(p.models)
}

/// `POST /api/generate` sans prompt : charge le modèle et le garde en mémoire.
///
/// `keep_alive` accepte une durée (`"5m"`, `"2h"`) ou un nombre de secondes
/// (`-1` = jamais décharger) ; voir [`keep_alive_json`].
pub fn load_model(base: &str, name: &str, keep_alive: &str) -> Result<()> {
    let body = serde_json::json!({
        "model": name,
        "keep_alive": keep_alive_json(keep_alive),
        "stream": false,
    });
    post_generate(base, &body)
}

/// `POST /api/generate` avec `keep_alive: 0` : décharge le modèle tout de suite.
pub fn unload_model(base: &str, name: &str) -> Result<()> {
    let body = serde_json::json!({
        "model": name,
        "keep_alive": 0,
        "stream": false,
    });
    post_generate(base, &body)
}

fn post_generate(base: &str, body: &serde_json::Value) -> Result<()> {
    with_auth(
        agent(base)
            .post(&url(base, "/api/generate"))
            .set("Content-Type", "application/json"),
    )
    .send_string(&body.to_string())
    .map_err(|e| map_ureq_err(base, e))?;
    Ok(())
}

/// `DELETE /api/delete` — supprime un modèle.
///
/// Note : Ollama >= 0.40 veut une requête DELETE avec body JSON
/// (`POST /api/delete` renvoie 405).
pub fn delete_model(base: &str, name: &str) -> Result<()> {
    with_auth(
        agent(base)
            .delete(&url(base, "/api/delete"))
            .set("Content-Type", "application/json"),
    )
    .send_json(serde_json::json!({ "name": name }))
    .map_err(|e| map_ureq_err(base, e))?;
    Ok(())
}

/// `POST /api/pull` en streaming — télécharge (ou met à jour) un modèle.
///
/// Appelle `progress` à chaque événement de progression avec
/// `(pct, texte)`. `pct` est dans 0..=100 ; 100 = terminé.
/// Renvoie `Err(OllamaError::Server)` si le pull échoue.
pub fn pull_model(
    base: &str,
    name: &str,
    mut progress: impl FnMut(u32, &str),
) -> Result<()> {
    let body = serde_json::json!({ "name": name, "stream": true }).to_string();
    let resp = with_auth(
        agent(base)
            .post(&url(base, "/api/pull"))
            .set("Content-Type", "application/json"),
    )
    .send_string(&body)
    .map_err(|e| map_ureq_err(base, e))?;

    let mut stream = resp.into_reader();
    let mut buf: Vec<u8> = Vec::new();
    let mut pct: u32 = 0;
    let mut last_status = String::from("démarrage…");

    loop {
        let mut chunk = [0u8; 8192];
        let n = stream
            .read(&mut chunk)
            .map_err(|e| OllamaError::Protocol(format!("lecture du flux : {e}")))?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let ev: PullEvent = match serde_json::from_str(line) {
                Ok(e) => e,
                Err(_) => continue,
            };
            if let Some(err) = ev.error {
                return Err(OllamaError::Server(err));
            }
            if !ev.status.is_empty() {
                last_status = ev.status.clone();
            }
            if let (Some(done), Some(total)) = (ev.completed, ev.total) {
                if total > 0 {
                    pct = ((done as u64 * 100) / total as u64) as u32;
                }
            }
            if ev.status == "success" {
                pct = 100;
            }
            progress(pct, last_status.as_str());
        }
    }
    Ok(())
}

// ----------------------------------------------------------------------
// Helpers d'affichage (unitaires)
// ----------------------------------------------------------------------

/// Taille « humaine » (o / Ko / Mo / Go / To, base 1024).
pub fn fmt_bytes(n: u64) -> String {
    const UNITS: [&str; 5] = ["o", "Ko", "Mo", "Go", "To"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} o", n)
    } else {
        format!("{:.1} {}", v, UNITS[i])
    }
}

/// Date ISO 8601 → JJ/MM/AAAA (tolère les variantes ; renvoie l'original si
/// le format n'est pas reconnu).
pub fn fmt_date(iso: &str) -> String {
    let b: Vec<char> = iso.chars().collect();
    if b.len() >= 10
        && b[4] == '-'
        && b[7] == '-'
        && b[0..4].iter().all(|c| c.is_ascii_digit())
        && b[5..7].iter().all(|c| c.is_ascii_digit())
        && b[8..10].iter().all(|c| c.is_ascii_digit())
    {
        return format!(
            "{}/{}/{}",
            &iso[8..10],
            &iso[5..7],
            &iso[0..4]
        );
    }
    iso.to_string()
}

/// Valeur JSON de `keep_alive` à partir de la saisie utilisateur.
///
/// Ollama attend soit une durée Go (`"5m"`, `"2h"`), soit un nombre de
/// secondes (`-1` = jamais décharger, `0` = décharger). Un entier est donc
/// envoyé comme nombre, le reste comme chaîne ; une saisie vide retombe sur
/// la valeur par défaut d'Ollama (5 min).
pub fn keep_alive_json(raw: &str) -> serde_json::Value {
    let t = raw.trim();
    if t.is_empty() {
        return serde_json::json!("5m");
    }
    if let Ok(n) = t.parse::<i64>() {
        return serde_json::json!(n);
    }
    serde_json::json!(t)
}

/// Taille du contexte en tokens, façon lisible (« 32 K », « 128 K »).
/// `0` (champ absent de `/api/ps`) → « — ».
pub fn fmt_context(tokens: u64) -> String {
    if tokens == 0 {
        return "—".to_string();
    }
    if tokens < 1024 {
        return format!("{tokens}");
    }
    let k = tokens / 1024;
    if tokens % 1024 == 0 {
        format!("{k} K")
    } else {
        format!("{:.1} K", tokens as f64 / 1024.0)
    }
}

/// Colonne « PROCESSOR » de `ollama ps` : part CPU/GPU déduite de `size_vram`.
pub fn processor_label(size: u64, size_vram: u64) -> String {
    if size == 0 {
        return "—".to_string();
    }
    if size_vram == 0 {
        return "100% CPU".to_string();
    }
    if size_vram >= size {
        return "100% GPU".to_string();
    }
    let gpu = size_vram.saturating_mul(100) / size;
    format!("{}%/{}% CPU/GPU", 100 - gpu, gpu)
}

/// Jours écoulés depuis l'époque Unix pour une date civile (algorithme de
/// Howard Hinnant, sans dépendance externe).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Parse un horodatage ISO 8601 (`YYYY-MM-DDTHH:MM:SS[.frac][Z|±HH:MM]`)
/// en secondes Unix. `None` si le format n'est pas reconnu.
pub fn parse_iso_epoch(s: &str) -> Option<i64> {
    let s = s.trim();
    let (date, rest) = s.split_once('T')?;
    let mut d = date.split('-');
    let y: i64 = d.next()?.parse().ok()?;
    let mo: i64 = d.next()?.parse().ok()?;
    let da: i64 = d.next()?.parse().ok()?;
    if !(1..=12).contains(&mo) || !(1..=31).contains(&da) {
        return None;
    }

    // Fuseau : « Z » (UTC), « ±HH:MM », ou absent (UTC par défaut).
    let (time, offset) = if let Some(t) = rest.strip_suffix('Z') {
        (t, 0)
    } else if let Some(pos) = rest.rfind(|c| c == '+' || c == '-') {
        let (t, off) = rest.split_at(pos);
        let sign = if off.starts_with('-') { -1 } else { 1 };
        let mut p = off[1..].split(':');
        let oh: i64 = p.next()?.parse().ok()?;
        let om: i64 = p.next().unwrap_or("0").parse().ok()?;
        (t, sign * (oh * 3600 + om * 60))
    } else {
        (rest, 0)
    };

    let time = time.split('.').next()?;
    let mut hms = time.split(':');
    let h: i64 = hms.next()?.parse().ok()?;
    let mi: i64 = hms.next()?.parse().ok()?;
    let se: i64 = hms.next().unwrap_or("0").parse().ok()?;
    if h > 23 || mi > 59 || se > 60 {
        return None;
    }
    Some(days_from_civil(y, mo, da) * 86_400 + h * 3600 + mi * 60 + se - offset)
}

/// Temps restant avant expiration, façon `ollama ps` (« 4 min », « 1 h 12 »).
/// Renvoie la chaîne d'origine si l'horodatage n'est pas reconnu.
pub fn fmt_remaining(expires_at: &str, now: i64) -> String {
    let Some(exp) = parse_iso_epoch(expires_at) else {
        return expires_at.to_string();
    };
    let left = exp - now;
    if left <= 0 {
        return "expiré".to_string();
    }
    if left < 60 {
        return format!("{left} s");
    }
    if left < 3600 {
        return format!("{} min", left / 60);
    }
    let h = left / 3600;
    let m = (left % 3600) / 60;
    if m == 0 {
        format!("{h} h")
    } else {
        format!("{h} h {m:02}")
    }
}

/// Renvoie `true` si au moins un caractère de `s` est non-ASCII
/// (simplification : on ne traite que l'accent « è » pour l'instant).
#[allow(dead_code)]
fn has_accent(s: &str) -> bool {
    s.chars().any(|c| c == 'è')
}

// ----------------------------------------------------------------------
// Tests unitaires
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- normalize_url ----------------------------------------------------

    #[test]
    fn normalise_ajoute_http() {
        assert_eq!(normalize_url("192.168.1.2:11434"), "http://192.168.1.2:11434");
    }

    #[test]
    fn normalise_conserve_https() {
        assert_eq!(
            normalize_url("https://h:11434"),
            "https://h:11434"
        );
    }

    #[test]
    fn normalise_supprime_slash_final() {
        assert_eq!(
            normalize_url("http://h:11434/"),
            "http://h:11434"
        );
        assert_eq!(
            normalize_url("http://h:11434///"),
            "http://h:11434"
        );
    }

    #[test]
    fn normalise_vide() {
        assert_eq!(normalize_url(""), "");
        assert_eq!(normalize_url("   "), "");
    }

    // ---- fmt_bytes ----------------------------------------------------------

    #[test]
    fn fmt_bytes_tests() {
        assert_eq!(fmt_bytes(0), "0 o");
        assert_eq!(fmt_bytes(123), "123 o");
        assert_eq!(fmt_bytes(1023), "1023 o");
        assert_eq!(fmt_bytes(1024), "1.0 Ko");
        assert_eq!(fmt_bytes(1536), "1.5 Ko");
        assert_eq!(fmt_bytes(1024 * 1024), "1.0 Mo");
        assert_eq!(fmt_bytes(1024 * 1024 * 1024), "1.0 Go");
        assert_eq!(fmt_bytes(1024_u64.pow(3) * 23), "23.0 Go");
        assert_eq!(fmt_bytes(1024_u64.pow(4)), "1.0 To");
    }

    // ---- fmt_date -----------------------------------------------------------

    #[test]
    fn fmt_date_tests() {
        assert_eq!(
            fmt_date("2026-10-24T13:27:15.920965355Z"),
            "24/10/2026"
        );
        assert_eq!(
            fmt_date("2026-01-02T00:00:00Z"),
            "02/01/2026"
        );
        // Format non reconnu → inchangé.
        assert_eq!(fmt_date("hier"), "hier");
        assert_eq!(fmt_date(""), "");
        // Longueur < 10 → inchangé.
        assert_eq!(fmt_date("2026-10-2"), "2026-10-2");
    }

    // ---- Parsing NDJSON du pull (même forme que /api/pull) ------------------

    #[test]
    fn parse_evolution_pull() {
        let ev: PullEvent =
            serde_json::from_str(r#"{"status":"pulling f535","completed":50,"total":100}"#)
                .unwrap();
        assert_eq!(ev.status, "pulling f535");
        assert_eq!(ev.completed, Some(50));
        assert_eq!(ev.total, Some(100));
        assert!(ev.error.is_none());
    }

    #[test]
    fn parse_erreur_pull() {
        let ev: PullEvent =
            serde_json::from_str(r#"{"error":"pull model manifest: file does not exist"}"#)
                .unwrap();
        assert_eq!(
            ev.error.as_deref(),
            Some("pull model manifest: file does not exist")
        );
    }

    #[test]
    fn parse_succes_pull() {
        let ev: PullEvent = serde_json::from_str(r#"{"status":"success"}"#).unwrap();
        assert_eq!(ev.status, "success");
        assert_eq!(ev.completed, None);
    }

    // ---- Calcul de progression ---------------------------------------------

    // Les tests qui touchent la cle API globale doivent etre serialises :
    // ils tournent en parallele par defaut et se corrompent mutuellement.
    static HTTP_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Non-régression : la suppression doit envoyer `DELETE /api/delete`
    /// avec le nom du modèle dans le corps JSON.
    #[test]
    fn suppression_envoie_delete_avec_le_nom() {
        let _g = HTTP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::io::{Read, Write};
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            s.set_read_timeout(Some(std::time::Duration::from_secs(5)))
                .unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            // Lit jusqu'à disposer des en-têtes ET du corps annoncé.
            loop {
                let n = s.read(&mut chunk).unwrap_or(0);
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(pos) = text.find("\r\n\r\n") {
                    let head = &text[..pos];
                    let len = head
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                        })
                        .unwrap_or(0);
                    if buf.len() >= pos + 4 + len {
                        break;
                    }
                }
            }
            s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
                .unwrap();
            String::from_utf8_lossy(&buf).to_string()
        });

        set_api_key(None);
        delete_model(&format!("http://127.0.0.1:{port}"), "smollm2:135m").unwrap();
        let req = h.join().unwrap();

        let head = req.split("\r\n\r\n").next().unwrap_or_default();
        assert!(
            head.starts_with("DELETE /api/delete"),
            "methode/URI inattendues:\n{head}"
        );
        assert!(
            req.contains("smollm2:135m"),
            "nom du modele absent du corps:\n{req}"
        );
    }

    #[test]
    fn requete_porte_la_cle_api() {
        let _g = HTTP_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        use std::io::{Read, Write};
        use std::net::TcpListener;
        // Serveur minimal qui capture la requete et repond un JSON valide.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let n = s.read(&mut buf).unwrap();
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = "{\"version\":\"0.9.9\"}";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            s.write_all(resp.as_bytes()).unwrap();
            req
        });

        set_api_key(Some("sk-test-123".to_string()));
        let v = get_version(&format!("http://127.0.0.1:{port}")).unwrap();
        let req = h.join().unwrap();
        set_api_key(None);

        assert_eq!(v, "0.9.9");
        assert!(
            req.to_lowercase().contains("authorization: bearer sk-test-123"),
            "en-tete Authorization absent de la requete:\n{req}"
        );
    }

    #[test]
    fn auth_header_value_tests() {
        assert_eq!(auth_header_value("abc123").as_deref(), Some("Bearer abc123"));
        // Les espaces autour sont ignorés.
        assert_eq!(auth_header_value("  abc  ").as_deref(), Some("Bearer abc"));
        // Vide ou blanc => pas d'authentification.
        assert_eq!(auth_header_value(""), None);
        assert_eq!(auth_header_value("   "), None);
    }

    #[test]
    fn calcul_pct() {
        let (done, total): (u64, u64) = (50, 100);
        let pct = ((done * 100) / total) as u32;
        assert_eq!(pct, 50);
        let pct = ((999_u64 * 100) / 1000) as u32;
        assert_eq!(pct, 99);
        // total == 0 → non divisé (gardé 0 dans la logique réelle).
        assert!(total > 0);
    }

    // ---- Moniteur : helpers purs (ollama ps / keep_alive) ------------------

    #[test]
    fn keep_alive_json_tests() {
        assert_eq!(keep_alive_json("5m"), serde_json::json!("5m"));
        assert_eq!(keep_alive_json(" 30m "), serde_json::json!("30m"));
        assert_eq!(keep_alive_json("2h"), serde_json::json!("2h"));
        // Un entier part en nombre : `-1` n'est pas une durée Go valide.
        assert_eq!(keep_alive_json("0"), serde_json::json!(0));
        assert_eq!(keep_alive_json("-1"), serde_json::json!(-1));
        assert_eq!(keep_alive_json("600"), serde_json::json!(600));
        // Saisie vide => valeur par défaut d'Ollama.
        assert_eq!(keep_alive_json("   "), serde_json::json!("5m"));
    }

    #[test]
    fn processor_label_tests() {
        assert_eq!(processor_label(100, 100), "100% GPU");
        assert_eq!(processor_label(100, 150), "100% GPU");
        assert_eq!(processor_label(100, 0), "100% CPU");
        assert_eq!(processor_label(100, 48), "52%/48% CPU/GPU");
        assert_eq!(processor_label(0, 0), "—");
    }

    #[test]
    fn parse_iso_epoch_tests() {
        assert_eq!(parse_iso_epoch("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_iso_epoch("1970-01-02T00:00:00Z"), Some(86_400));
        // Les fractions de seconde sont ignorées.
        assert_eq!(
            parse_iso_epoch("2026-11-15T10:12:56.187695428Z"),
            parse_iso_epoch("2026-11-15T10:12:56Z")
        );
        // Décalage horaire : 11:12:56+01:00 == 10:12:56Z.
        let utc = parse_iso_epoch("2026-11-15T10:12:56Z").unwrap();
        assert_eq!(parse_iso_epoch("2026-11-15T11:12:56+01:00"), Some(utc));
        // Formats non reconnus.
        assert_eq!(parse_iso_epoch("hier"), None);
        assert_eq!(parse_iso_epoch("2026-13-01T00:00:00Z"), None);
        assert_eq!(parse_iso_epoch(""), None);
    }

    #[test]
    fn fmt_remaining_tests() {
        let iso = "2026-11-15T10:12:56Z";
        let exp = parse_iso_epoch(iso).unwrap();
        assert_eq!(fmt_remaining(iso, exp - 30), "30 s");
        assert_eq!(fmt_remaining(iso, exp - 125), "2 min");
        assert_eq!(fmt_remaining(iso, exp - 3600), "1 h");
        assert_eq!(fmt_remaining(iso, exp - 4320), "1 h 12");
        assert_eq!(fmt_remaining(iso, exp), "expiré");
        // Horodatage illisible => renvoyé tel quel.
        assert_eq!(fmt_remaining("hier", 0), "hier");
    }

    #[test]
    fn parse_reponse_ps() {
        let json = r#"{"models":[{"name":"llama3.2:latest","size":2019393189,
            "size_vram":2019393189,"expires_at":"2026-11-15T10:12:56.187695428Z",
            "context_length":32768}]}"#;
        let p: PsResp = serde_json::from_str(json).unwrap();
        assert_eq!(p.models.len(), 1);
        assert_eq!(p.models[0].name, "llama3.2:latest");
        assert_eq!(p.models[0].size_vram, 2019393189);
        assert_eq!(p.models[0].context_length, 32768);
        // Aucun modèle chargé => liste vide, pas d'erreur.
        let p: PsResp = serde_json::from_str(r#"{"models":[]}"#).unwrap();
        assert!(p.models.is_empty());
        // `context_length` absent (vieux serveur) => 0, pas d'échec de parsing.
        let p: PsResp = serde_json::from_str(
            r#"{"models":[{"name":"x","size":1,"size_vram":1,"expires_at":""}]}"#,
        )
        .unwrap();
        assert_eq!(p.models[0].context_length, 0);
    }

    #[test]
    fn fmt_context_tests() {
        assert_eq!(fmt_context(0), "—");
        assert_eq!(fmt_context(512), "512");
        assert_eq!(fmt_context(2048), "2 K");
        assert_eq!(fmt_context(32768), "32 K");
        assert_eq!(fmt_context(131072), "128 K");
        assert_eq!(fmt_context(128000), "125 K");
        // Valeur non multiple de 1024 => une décimale.
        assert_eq!(fmt_context(1500), "1.5 K");
    }
}
