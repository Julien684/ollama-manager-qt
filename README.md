# ollama-manager-qt

Gestionnaire graphique **Qt6** d'Ollama — binaire unique en **Rust** (via la
crate [`qtrs`](https://github.com/mateuszwq/qtrs)).

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
  https://depoa.julien68.fr/Linuxon/ollama-manager-qt/raw/branch/main/ollama-manager-qt-0.1.0-1-1-x86_64.eopkg
```

Ou depuis le dépôt binaire personnel : `sudo eopkg install -y ollama-manager-qt`
(dépôt `linuxon` — voir `Linuxon/eopkg-repo`). Seul prérequis : `qt6-base`
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
