# ollama-manager-qt

**Ollama Model Manager** — gestionnaire graphique **Qt6** d'Ollama, binaire
unique en **Rust** (via la crate [`qtrs`](https://github.com/mateuszwq/qtrs)).

Télécharge, met à jour et supprime des modèles Ollama depuis une interface
native. L'adresse du serveur est configurable au lancement et **persistée**
(`QSettings`, `~/.config/OllamaManager/ollama-manager-qt.ini`).

## Fonctionnalités

- **Configuration du serveur** — champ d'adresse (ex.
  `http://192.168.1.2:11434`), bouton *Tester* (version Ollama) et
  *Enregistrer* (persistance QSettings + rechargement de la liste).
- **Téléchargement** — saisie d'un nom de modèle (ex. `llama3.2`,
  `qwen2.5:7b`, `hf.co/nom/depot:tag`), barre de progression en direct
  (stream NDJSON `/api/pull`).
- **Modèles locaux** — table (nom, taille, paramètres, date, capacités) avec
  :
  - *Actualiser*
  - *Mettre à jour* (ligne sélectionnée)
  - *Tout mettre à jour* (séquencé)
  - *Supprimer* (confirmation modale, `DELETE /api/delete`)

## Construction

```bash
# Prérequis : Rust (>= 1.85) + Qt6 (libs dev: QtCore QtGui QtWidgets)
cargo build --release
# → target/release/ollama-manager-qt  (binaire unique, ~3 Mo)
```

## Installation (Solus)

Paquet `.eopkg` fourni dans ce dépôt :

```bash
sudo eopkg install -y \
  https://raw.githubusercontent.com/Julien684/ollama-manager-qt/main/ollama-manager-qt-0.1.0-11-1-x86_64.eopkg
```

Ou depuis le dépôt binaire personnel : `sudo eopkg install -y ollama-manager-qt`
(dépôt `linuxon` — voir `Julien684/eopkg-repo`). Seul prérequis : `qt6-base`
(fourni par le système Solus).

## Tests

```bash
cargo test
```

Les tests unitaires couvrent la couche client Ollama
(`src/ollama.rs`) : formatage (tailles, dates), analyse JSON des réponses
API, construction des URL, mappage d'erreurs — sans ouvrir de fenêtre ni
contacter un serveur.

## Architecture

| Fichier | Rôle |
|---------|------|
| `src/main.rs` | GUI Qt6 (fenêtre, layouts, callbacks), « pompe » `mpsc` → widgets via `qtrs::Timer` |
| `src/ollama.rs` | Client HTTP Ollama (version, tags, pull stream, delete), testable unitairement |

Pattern d'ownership `qtrs` : les widgets interactifs (table, barre, statut)
sont en `Rc<RefCell<..>>` + `set_has_parent()` ; le travail long (HTTP) tourne
en `std::thread` et renvoie ses résultats par canaux `std::sync::mpsc`,
drainés par un `Timer` Qt à 200 ms dans le thread UI.

## API Ollama utilisée

- `GET /api/version`
- `GET /api/tags`
- `POST /api/pull` (stream NDJSON)
- `DELETE /api/delete` (body JSON `{"name": "..."}`)

## Licence

GPL-3.0-or-later

---

# Ollama Model Manager (English)

**Ollama Model Manager** — a graphical **Qt6** manager for Ollama, a single
binary written in **Rust** (via the [`qtrs`](https://github.com/mateuszwq/qtrs) crate).

Download, update and remove Ollama models from a native interface. The server
address is configurable at startup and **persisted**
(`QSettings`, `~/.config/OllamaManager/ollama-manager-qt.ini`).

## Features

- **Server configuration** — address field (e.g.
  `http://192.168.1.2:11434`), a *Test* button (Ollama version) and a
  *Save* button (QSettings persistence + list reload).
- **Download** — model name input (e.g. `llama3.2`, `qwen2.5:7b`,
  `hf.co/user/repo:tag`), live progress bar (NDJSON stream `/api/pull`).
- **Local models** — table (name, size, parameters, date, capabilities) with:
  - *Refresh*
  - *Update* (selected row)
  - *Update all* (sequential)
  - *Delete* (modal confirmation, `DELETE /api/delete`)

## Building

```bash
# Prerequisites: Rust (>= 1.85) + Qt6 (dev libs: QtCore QtGui QtWidgets)
cargo build --release
# → target/release/ollama-manager-qt  (single binary, ~3 MB)
```

## Installation (Solus)

The `.eopkg` package is provided in this repository:

```bash
sudo eopkg install -y \
  https://raw.githubusercontent.com/Julien684/ollama-manager-qt/main/ollama-manager-qt-0.1.0-11-1-x86_64.eopkg
```

Or from the personal binary repository: `sudo eopkg install -y ollama-manager-qt`
(the `linuxon` repo — see `Julien684/eopkg-repo`). Only prerequisite: `qt6-base`
(provided by the Solus system).

## Tests

```bash
cargo test
```

Unit tests cover the Ollama client layer
(`src/ollama.rs`): formatting (sizes, dates), parsing of API JSON responses,
URL building, error mapping — no window opened and no server contacted.

## Architecture

| File | Role |
|------|------|
| `src/main.rs` | Qt6 GUI (window, layouts, callbacks), `mpsc` "pump" → widgets via `qtrs::Timer` |
| `src/ollama.rs` | Ollama HTTP client (version, tags, pull stream, delete), unit-testable |

`qtrs` ownership pattern: the interactive widgets (table, progress bar,
status) live in `Rc<RefCell<..>>` + `set_has_parent()`; the long-running work
(HTTP) runs in a `std::thread` and returns results through `std::sync::mpsc`
channels, drained by a 200 ms Qt `Timer` in the UI thread.

## Ollama API endpoints used

- `GET /api/version`
- `GET /api/tags`
- `POST /api/pull` (NDJSON stream)
- `DELETE /api/delete` (JSON body `{"name": "..."}`)

## License

GPL-3.0-or-later
