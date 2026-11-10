#!/usr/bin/env python3
"""
build_deb.py — Construit un paquet Debian/Ubuntu (.deb) pour ollama-manager-qt.

Usage :
    python3 build_deb.py [repertoire_source] [repertoire_sortie]

Le binaire doit avoir ete compile avant (cargo build --release).
Necessite dpkg-deb (paquet dpkg-dev / dpkg).

Le paquet installe les memes fichiers que le .eopkg Solus :
    /usr/bin/ollama-manager-qt
    /usr/share/applications/ollama-manager-qt.desktop
    /usr/share/pixmaps/ollama-manager-qt.png
    /usr/share/icons/hicolor/<taille>/apps/ollama-manager-qt.png
    /usr/share/doc/ollama-manager-qt/{README.md,screenshot.png,LICENSE}
"""
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

SRC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else SRC_DIR

PACKAGE_NAME = "ollama-manager-qt"
VERSION = "0.1.0"
DEB_REVISION = 12
ARCH = "amd64"
MAINTAINER = "Julien <julien_solus@use.startmail.com>"
HOMEPAGE = "https://github.com/Julien684/ollama-manager-qt"
DESCRIPTION = (
    "Ollama Model Manager - graphical Qt6 manager for Ollama\n"
    " Single-binary Rust/Qt6 application to download, update and remove\n"
    " Ollama models, with a live progress bar, a local-models table and a\n"
    " configurable, persisted server address."
)

# Alternatives : Debian trixie (et Ubuntu recent) a renomme les paquets Qt6
# avec le suffixe t64 (transition time64_t).
DEPENDS = ", ".join([
    "libc6",
    "libstdc++6",
    "libgcc-s1",
    "libqt6core6t64 | libqt6core6",
    "libqt6gui6t64 | libqt6gui6",
    "libqt6widgets6t64 | libqt6widgets6",
    "libqt6dbus6t64 | libqt6dbus6",
])

FILES = [
    ("usr/bin/ollama-manager-qt", "target/release/ollama-manager-qt", 0o755),
    ("usr/share/applications/ollama-manager-qt.desktop",
     "ollama-manager-qt.desktop", 0o644),
    ("usr/share/pixmaps/ollama-manager-qt.png",
     "icons/ollama-manager-qt-256.png", 0o644),
    ("usr/share/doc/ollama-manager-qt/README.md", "README.md", 0o644),
    ("usr/share/doc/ollama-manager-qt/screenshot.png", "screenshot.png", 0o644),
    ("usr/share/doc/ollama-manager-qt/LICENSE", "LICENSE", 0o644),
]
for px in (24, 32, 48, 64, 128, 256):
    FILES.append((
        f"usr/share/icons/hicolor/{px}x{px}/apps/ollama-manager-qt.png",
        f"icons/ollama-manager-qt-{px}.png", 0o644,
    ))

OUT_FILENAME = f"{PACKAGE_NAME}_{VERSION}-{DEB_REVISION}_{ARCH}.deb"


def build() -> str:
    staging = OUT_DIR / f".deb-staging-{PACKAGE_NAME}"
    if staging.exists():
        shutil.rmtree(staging)

    for dest, src_rel, mode in FILES:
        src = SRC_DIR / src_rel
        if not src.exists():
            raise SystemExit(f"Fichier source absent : {src}")
        dst = staging / dest
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        os.chmod(dst, mode)

    debian = staging / "DEBIAN"
    debian.mkdir(parents=True)
    installed_size = sum(
        (staging / d).stat().st_size for d, _, _ in FILES
    ) // 1024
    control = f"""Package: {PACKAGE_NAME}
Version: {VERSION}-{DEB_REVISION}
Section: network
Priority: optional
Architecture: {ARCH}
Depends: {DEPENDS}
Installed-Size: {installed_size}
Maintainer: {MAINTAINER}
Homepage: {HOMEPAGE}
Description: {DESCRIPTION}
"""
    (debian / "control").write_text(control, encoding="utf-8")

    out = OUT_DIR / OUT_FILENAME
    subprocess.run(
        ["dpkg-deb", "--build", "--root-owner-group", str(staging), str(out)],
        check=True,
    )
    shutil.rmtree(staging)
    print(f"\n  OK {out}  ({out.stat().st_size} octets)")
    return str(out)


if __name__ == "__main__":
    build()
