#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="$REPO_ROOT/.venvs/download"
RUNTIME_DIR="$VENV_DIR/.runtime"
BIN_DIR="$VENV_DIR/bin"
DENO_CACHE_DIR="$REPO_ROOT/.data/s1-download/deno"
PY="$VENV_DIR/bin/python"

usage() {
    echo "Usage: $0 [--force|--clean|--recreate]"
    echo "Provision .venvs/download with Python, current yt-dlp, and local JavaScript runtimes."
}

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force|--clean|--recreate) FORCE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "Missing uv; install uv before provisioning the download environment." >&2
    exit 2
fi
if ! command -v curl >/dev/null 2>&1; then
    echo "Missing curl; it is required to install the JavaScript runtimes." >&2
    exit 2
fi
if ! command -v tar >/dev/null 2>&1; then
    echo "Missing tar; it is required to install Node.js." >&2
    exit 2
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "Missing unzip; it is required to install Bun." >&2
    exit 2
fi

if [ "$FORCE" -eq 1 ] && [ -d "$VENV_DIR" ]; then
    echo "🗑️  Removing existing ${VENV_DIR}..."
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating Python 3.13 virtual environment ${VENV_DIR}..."
    uv venv --python 3.13 "$VENV_DIR"
fi

mkdir -p "$RUNTIME_DIR" "$BIN_DIR" "$DENO_CACHE_DIR"

install_deno() {
    echo "📦 Installing/updating Deno in ${BIN_DIR}..."
    # DENO_INSTALL makes the official installer project-local. The download
    # launcher adds this bin directory to PATH for yt-dlp.
    curl -fsSL https://deno.land/install.sh | DENO_INSTALL="$VENV_DIR" sh
}

install_node() {
    local node_arch node_tag node_archive node_root temp_dir
    case "$(uname -m)" in
        x86_64|amd64) node_arch="x64" ;;
        aarch64|arm64) node_arch="arm64" ;;
        *)
            echo "Unsupported Linux architecture for the Node.js bundle: $(uname -m)" >&2
            exit 1
            ;;
    esac

    node_tag=$(curl -fsSL https://nodejs.org/dist/index.json | "$PY" -c '
import json
import sys

releases = json.load(sys.stdin)
for release in releases:
    if release.get("lts"):
        print(release["version"])
        break
else:
    raise SystemExit("Node.js LTS release was not found")
')
    node_archive="node-${node_tag}-linux-${node_arch}.tar.xz"
    node_root="$RUNTIME_DIR/node"
    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' RETURN
    echo "📦 Installing Node.js ${node_tag} (LTS) in ${node_root}..."
    curl -fsSL "https://nodejs.org/dist/${node_tag}/${node_archive}" -o "$temp_dir/$node_archive"
    tar -xJf "$temp_dir/$node_archive" -C "$temp_dir"
    rm -rf "$node_root"
    mv "$temp_dir/node-${node_tag}-linux-${node_arch}" "$node_root"
    ln -sfn "$node_root/bin/node" "$BIN_DIR/node"
    ln -sfn "$node_root/bin/npm" "$BIN_DIR/npm"
    ln -sfn "$node_root/bin/npx" "$BIN_DIR/npx"
    ln -sfn "$node_root/bin/corepack" "$BIN_DIR/corepack"
    trap - RETURN
    rm -rf "$temp_dir"
}

install_bun() {
    local bun_root="$RUNTIME_DIR/bun"
    # yt-dlp currently supports Bun through 1.3.14; newer Bun releases are
    # not selected because upstream support is deprecated.
    local bun_version="${BUN_VERSION:-bun-v1.3.14}"
    echo "📦 Installing Bun ${bun_version} in ${bun_root}..."
    rm -rf "$bun_root"
    curl -fsSL https://bun.com/install | BUN_INSTALL="$bun_root" bash -s "$bun_version"
    ln -sfn "$bun_root/bin/bun" "$BIN_DIR/bun"
}

install_quickjs() {
    local quickjs_archive quickjs_root temp_dir
    case "$(uname -m)" in
        x86_64|amd64)
            quickjs_archive=$(curl -fsSL https://bellard.org/quickjs/binary_releases/ \
                | "$PY" -c '
import re
import sys

names = re.findall(r"quickjs-linux-x86_64-[0-9-]+\.zip", sys.stdin.read())
if names:
    print(sorted(set(names))[-1])
')
            ;;
        *)
            echo "ℹ️  Skipping QuickJS: no upstream Linux binary is configured for $(uname -m)."
            return
            ;;
    esac
    if [ -z "$quickjs_archive" ]; then
        echo "Could not find an upstream QuickJS Linux x86_64 binary." >&2
        exit 1
    fi

    quickjs_root="$RUNTIME_DIR/quickjs"
    temp_dir=$(mktemp -d)
    trap 'rm -rf "$temp_dir"' RETURN
    echo "📦 Installing QuickJS (${quickjs_archive}) in ${quickjs_root}..."
    curl -fsSL "https://bellard.org/quickjs/binary_releases/${quickjs_archive}" -o "$temp_dir/quickjs.zip"
    "$PY" -c '
import pathlib
import sys
import zipfile

archive = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(archive) as source:
    for executable in ("qjs", "qjsc"):
        matches = [name for name in source.namelist() if name == executable or name.endswith("/" + executable)]
        if matches:
            target = destination / executable
            target.write_bytes(source.read(matches[0]))
            target.chmod(0o755)
' "$temp_dir/quickjs.zip" "$quickjs_root"
    ln -sfn "$quickjs_root/qjs" "$BIN_DIR/qjs"
    trap - RETURN
    rm -rf "$temp_dir"
}

echo "📦 Updating Python download requirements (including yt-dlp)..."
uv pip install --upgrade --python "$PY" -r "$REPO_ROOT/envs/requirements-download.txt"

install_deno
install_node
install_bun
install_quickjs

echo "✅ Verifying download environment..."
PATH="$BIN_DIR:$PATH" DENO_DIR="$DENO_CACHE_DIR" "$PY" -c '
import shutil
import sys
from importlib.metadata import version

print(f"   -> Python: {sys.version.split()[0]}")
yt_dlp_version = version("yt-dlp")
print(f"   -> yt-dlp: {yt_dlp_version}")
for executable in ("deno", "node", "npm", "bun", "qjs"):
    path = shutil.which(executable)
    print(f"   -> {executable}: {path or 'missing'}")
    if path is None:
        raise SystemExit(f"Missing JavaScript runtime: {executable}")
'

ln -sfn "$VENV_DIR" "$REPO_ROOT/.venv-download"
echo "🎉 ${VENV_DIR} ready!"
