#!/usr/bin/env python3
"""
build_eopkg.py — Builds a .eopkg for ollama-manager-qt in the exact Solus
binary format.

The output is a ZIP containing:
  - metadata.xml
  - files.xml
  - install.tar.xz

No COMAR scripts (pure file package, no install/uninstall hooks needed).
No <RuntimeDependencies>: `eopkg install <url>` downloads a single file and
cannot resolve deps against the Solus repo database. The only prerequisite is
qt6-base (system-provided on any Solus desktop), documented in README.md.
"""
import hashlib
import io
import os
import sys
import tarfile
import zipfile
from datetime import date
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC_DIR

PACKAGE_NAME   = "ollama-manager-qt"
VERSION        = "0.1.0"
RELEASE        = 4
DIST_RELEASE   = 1
ARCH           = "x86_64"
DIST_NAME      = "Solus"
BUILD_HOST     = "solus-local"
HOME_PAGE      = "https://depoa.julien68.fr/Linuxon/ollama-manager-qt"
PACKAGER_NAME  = "Julien"
PACKAGER_EMAIL = "julien_solus@use.startmail.com"
PART_OF        = "network.utilities"

# (dest_path, src_rel_path, mode)
# On Solus, /bin is a symlink to /usr/bin, so install to usr/bin.
FILE_LIST = [
    ("usr/bin/ollama-manager-qt",
     "target/release/ollama-manager-qt", 0o755),
    ("usr/share/applications/ollama-manager-qt.desktop",
     "ollama-manager-qt.desktop", 0o644),
    # Copie hors theme : le .desktop pointe ce chemin absolu, que les menus
    # (KDE/GTK) resolvent toujours, contrairement au seul nom de theme.
    ("usr/share/pixmaps/ollama-manager-qt.png",
     "icons/ollama-manager-qt-256.png", 0o644),
    ("usr/share/doc/ollama-manager-qt/README.md",
     "README.md", 0o644),
    ("usr/share/licenses/ollama-manager-qt/LICENSE",
     "LICENSE", 0o644),
]
# Icones hicolor : le .desktop pointe sur « ollama-manager-qt », le thème
# hicolor fournit toutes les tailles.
for _px in (24, 32, 48, 64, 128, 256):
    FILE_LIST.append((
        f"usr/share/icons/hicolor/{_px}x{_px}/apps/ollama-manager-qt.png",
        f"icons/ollama-manager-qt-{_px}.png", 0o644,
    ))

# Runtime dependencies: qt6-base is system-provided on any Solus desktop and
# `eopkg install <url>` cannot resolve deps anyway — documented in README.
RUNTIME_DEPS = []

OUT_FILENAME = f"{PACKAGE_NAME}-{VERSION}-{RELEASE}-{DIST_RELEASE}-{ARCH}.eopkg"

# French metadata — accents preserved in UTF-8 (must NOT pass through the
# ASCII-replacing helper `a()`).
_FR_SUMMARY = ("Gestionnaire graphique (Qt6) d'Ollama - telecharger, mettre a "
               "jour et supprimer des modeles")
_FR_DESCRIPTION = """ollama-manager-qt est un petit gestionnaire graphique (Qt6), ecrit en
Rust, pour administrer un serveur Ollama : telechargement de modeles avec
barre de progression en direct, mise a jour d'un modele ou de tous les
modeaux, suppression avec confirmation, et liste des modeaux locaux (nom,
taille, parametres, date, capacites). L'adresse du serveur est configurable
au lancement et persistee dans QSettings. Binaire unique (~3 Mo) sans
dependance runtime a part qt6-base (fourni par le systeme). Non officiel."""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def sha1_hex(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def file_type_for(path: str) -> str:
    """Return the eopkg file type for a given path (no leading /)."""
    if path.startswith("usr/bin/") or path.startswith("bin/"):
        return "executable"
    if path.startswith("usr/libexec/") or path.startswith("usr/sbin/") or path.startswith("sbin/"):
        return "executable"
    if path.startswith("usr/include/"):
        return "header"
    if path.startswith("usr/lib/"):
        return "library"
    if path.startswith("usr/share/man/"):
        return "man"
    if path.startswith("usr/share/info/"):
        return "info"
    if path.startswith("usr/share/locale/"):
        return "localedata"
    if path.startswith("usr/share/doc/") or path.startswith("usr/share/gtk-doc/") or path.startswith("usr/share/help/"):
        return "doc"
    if path.startswith("usr/lib/pkgconfig/") or path.startswith("usr/lib32/pkgconfig/") or path.startswith("usr/lib64/pkgconfig/"):
        return "data"
    if path.startswith("etc/"):
        return "config"
    return "data"


# ──────────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────────
def build():
    out_path = OUT_DIR / OUT_FILENAME
    print(f"Building: {out_path}")

    file_entries = []  # (dest_path, bytes, mode)
    for dest_path, src_rel, mode in FILE_LIST:
        src = SRC_DIR / src_rel
        if not src.exists():
            print(f"  WARNING: source file not found: {src}", file=sys.stderr)
            continue
        data = src.read_bytes()
        file_entries.append((dest_path, data, mode))
        print(f"  + {dest_path}  ({len(data)} bytes, mode {oct(mode)})")

    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:xz") as tar:
        for dest_path, data, mode in file_entries:
            info = tarfile.TarInfo(name=dest_path)
            info.size = len(data)
            info.mode = mode
            info.uid = 0
            info.gid = 0
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    install_tar_xz = tar_buf.getvalue()
    print(f"  install.tar.xz: {len(install_tar_xz)} bytes")

    files_parts = ['<Files>']
    for dest_path, data, mode in file_entries:
        ftype = file_type_for(dest_path)
        mode_str = f"{mode:04o}"
        h = sha1_hex(data)
        files_parts.append("  <File>")
        files_parts.append(f"    <Path>{dest_path}</Path>")
        files_parts.append(f"    <Type>{ftype}</Type>")
        files_parts.append(f"    <Size>{len(data)}</Size>")
        files_parts.append(f"    <Uid>0</Uid>")
        files_parts.append(f"    <Gid>0</Gid>")
        files_parts.append(f"    <Mode>{mode_str}</Mode>")
        files_parts.append(f"    <Hash>{h}</Hash>")
        files_parts.append("  </File>")
    files_parts.append("</Files>")
    files_xml = "\n".join(files_parts).encode("utf-8")
    print(f"  files.xml: {len(files_xml)} bytes")

    today = date.today().isoformat()
    installed_size = sum(len(data) for _, data, _ in file_entries)

    deps_xml = ""
    if RUNTIME_DEPS:
        deps_xml = "    <RuntimeDependencies>\n"
        for dep in RUNTIME_DEPS:
            deps_xml += f"        <Dependency>{dep}</Dependency>\n"
        deps_xml += "    </RuntimeDependencies>\n"

    def a(s: str) -> str:
        return s.encode("ascii", "replace").decode("ascii")

    metadata_xml = f"""<PISI>
    <Source>
        <Name>{PACKAGE_NAME}</Name>
        <Homepage>{HOME_PAGE}</Homepage>
        <Packager>
            <Name>{PACKAGER_NAME}</Name>
            <Email>{PACKAGER_EMAIL}</Email>
        </Packager>
    </Source>
    <Package>
        <Name>{PACKAGE_NAME}</Name>
        <Summary xml:lang="en">{a("Graphical Qt6 manager for Ollama - download, update and remove models")}</Summary>
        <Description xml:lang="en">{a("""ollama-manager-qt is a small graphical Qt6 manager, written in Rust
(via qtrs), for administering an Ollama server: model download with a live
progress bar, updating one model or all of them, removal with confirmation,
and a table of local models (name, size, parameters, date, capabilities).
The server address is configurable at startup and persisted in QSettings.
Single binary (~3 MB) with no runtime dependency beyond qt6-base
(system-provided). Unofficial.""")}</Description>
        <Summary xml:lang="fr">{_FR_SUMMARY}</Summary>
        <Description xml:lang="fr">{_FR_DESCRIPTION}</Description>
        <PartOf>{PART_OF}</PartOf>
        <License>GPL-3.0-or-later</License>
{deps_xml}        <History>
            <Update release="{RELEASE}">
                <Date>{today}</Date>
                <Version>{VERSION}</Version>
                <Comment>{PACKAGE_NAME} {VERSION} : icone resolue par le menu (icone installee dans /usr/share/pixmaps + Icon= absolu, set_icon/set_desktop_file_name cote app), zone barre de progression + logs limitee en hauteur</Comment>
                <Name>{PACKAGER_NAME}</Name>
                <Email>{PACKAGER_EMAIL}</Email>
            </Update>
            <Update release="3">
                <Date>2026-11-01</Date>
                <Version>{VERSION}</Version>
                <Comment>{PACKAGE_NAME} {VERSION} : champ de telechargement reellement visible (GridLayout : libelle + champ + bouton), icone lama blanc avec cle a molette</Comment>
                <Name>{PACKAGER_NAME}</Name>
                <Email>{PACKAGER_EMAIL}</Email>
            </Update>
            <Update release="2">
                <Date>2026-11-01</Date>
                <Version>{VERSION}</Version>
                <Comment>{PACKAGE_NAME} {VERSION} : correctifs - SIGSEGV a la fermeture, timer de pompage, libelle du champ serveur, persistance config INI</Comment>
                <Name>{PACKAGER_NAME}</Name>
                <Email>{PACKAGER_EMAIL}</Email>
            </Update>
            <Update release="1">
                <Date>2026-11-01</Date>
                <Version>{VERSION}</Version>
                <Comment>{PACKAGE_NAME} {VERSION} : version initiale - GUI Qt6 (téléchargement / mise à jour / suppression de modèles Ollama, adresse serveur persistée)</Comment>
                <Name>{PACKAGER_NAME}</Name>
                <Email>{PACKAGER_EMAIL}</Email>
            </Update>
        </History>
        <BuildHost>{BUILD_HOST}</BuildHost>
        <Distribution>{DIST_NAME}</Distribution>
        <DistributionRelease>{DIST_RELEASE}</DistributionRelease>
        <Architecture>{ARCH}</Architecture>
        <InstalledSize>{installed_size}</InstalledSize>
        <PackageFormat>1.2</PackageFormat>
        <Source>
            <Name>{PACKAGE_NAME}</Name>
            <Homepage>{HOME_PAGE}</Homepage>
            <Packager>
                <Name>{PACKAGER_NAME}</Name>
                <Email>{PACKAGER_EMAIL}</Email>
            </Packager>
        </Source>
    </Package>
</PISI>""".encode("utf-8")
    print(f"  metadata.xml: {len(metadata_xml)} bytes")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.xml", metadata_xml)
        zf.writestr("files.xml", files_xml)
        zf.writestr("install.tar.xz", install_tar_xz)

    size = out_path.stat().st_size
    print(f"\n  ✓ {out_path}  ({size} bytes)")
    print(f"  SHA1: {sha1_hex(out_path.read_bytes())}")
    return str(out_path)


if __name__ == "__main__":
    build()
