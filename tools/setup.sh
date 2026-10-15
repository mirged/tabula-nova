#!/usr/bin/env bash
set -euo pipefail
NOVA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NOVA_BACKEND="${1:-cpu}"
case "${NOVA_BACKEND}" in
    cpu) NOVA_WHEEL_INDEX='https://download.pytorch.org/whl/cpu'; NOVA_TORCH_VERSION='2.8.0+cpu' ;;
    cuda) NOVA_WHEEL_INDEX='https://download.pytorch.org/whl/cu128'; NOVA_TORCH_VERSION='2.8.0+cu128' ;;
    *) echo 'Usage: bash tools/setup.sh [cpu|cuda]' >&2; exit 2 ;;
esac
if [[ ! -x "${NOVA_ROOT}/.venv/bin/python" ]]; then
    python3 -m venv --without-pip "${NOVA_ROOT}/.venv"
fi
NOVA_PYTHON="${NOVA_ROOT}/.venv/bin/python"
if ! "${NOVA_PYTHON}" -m pip --version >/dev/null 2>&1; then
    NOVA_BOOTSTRAP="$(mktemp -d)"
    trap 'rm -rf -- "${NOVA_BOOTSTRAP}"' EXIT
    curl -fsSL --max-time 60 https://bootstrap.pypa.io/get-pip.py -o "${NOVA_BOOTSTRAP}/get-pip.py"
    "${NOVA_PYTHON}" "${NOVA_BOOTSTRAP}/get-pip.py"
fi
"${NOVA_PYTHON}" -m pip install "torch==${NOVA_TORCH_VERSION}" --index-url "${NOVA_WHEEL_INDEX}"
"${NOVA_PYTHON}" -m pip install -e "${NOVA_ROOT}[data,oci]"
mkdir -p "${NOVA_ROOT}/reports"
"${NOVA_PYTHON}" -m pip freeze > "${NOVA_ROOT}/reports/environment-lock.txt"
"${NOVA_PYTHON}" -m pip check
cd -- "${NOVA_ROOT}"
"${NOVA_PYTHON}" -m tabula_nova.workspace doctor
