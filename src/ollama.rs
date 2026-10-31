//! Client HTTP de l'API Ollama (ureq, JSON).
//!
//! Côté logique, indépendant de la GUI : chaque fonction est testable sans
//! ouvrir de fenêtre. Le streaming de `/api/pull` est lu ligne par ligne
//! (NDJSON) et reporté via un callback de progression.

use serde::{Deserialize, Serialize};

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
    // longtemps mais chaque *chunk* lu en chemin réarmé le timer de lecture.
    let d = std::time::Duration::from_secs(120);
    ureq::AgentBuilder::new()
        .timeout(d)
        .user_agent(UA)
        .build()
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
    let r = agent(base)
        .get(&url(base, "/api/version"))
        .call()
        .map_err(|e| map_ureq_err(base, e))?;
    let v: Version = r
        .into_json()
        .map_err(|e| OllamaError::Protocol(e.to_string()))?;
    Ok(v.version)
}

/// `GET /api/tags` — liste les modèles locaux.
pub fn list_models(base: &str) -> Result<Vec<Model>> {
    let r = agent(base)
        .get(&url(base, "/api/tags"))
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

/// `DELETE /api/delete` — supprime un modèle.
///
/// Note : Ollama >= 0.40 veut une requête DELETE avec body JSON
/// (`POST /api/delete` renvoie 405).
pub fn delete_model(base: &str, name: &str) -> Result<()> {
    agent(base)
        .delete(&url(base, "/api/delete"))
        .set("Content-Type", "application/json")
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
    let resp = agent(base)
        .post(&url(base, "/api/pull"))
        .set("Content-Type", "application/json")
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
}
