#!/usr/bin/env bash
set -euo pipefail
NOVA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
NOVA_PYTHON="${NOVA_ROOT}/.venv/bin/python"
export PYTHONPATH="${NOVA_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${NOVA_ROOT}/data/cache/huggingface"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
cd -- "${NOVA_ROOT}"
NOVA_COMMAND="${1:-help}"
if (( $# )); then shift; fi
if [[ "${NOVA_COMMAND}" == setup ]]; then
    exec bash tools/setup.sh "$@"
fi
if [[ "${NOVA_COMMAND}" == help ]]; then
    cat <<'HELP'
Run one stage at a time from the project directory:
  bash tools/workflow.sh setup [cpu|cuda]
  bash tools/workflow.sh doctor
  bash tools/workflow.sh test
  bash tools/workflow.sh collect --documents 1000
  bash tools/workflow.sh synthetic --documents 1000
  bash tools/workflow.sh clean
  bash tools/workflow.sh mix
  bash tools/workflow.sh prepare
  bash tools/workflow.sh smoke
  bash tools/workflow.sh status
  bash tools/workflow.sh prepare-pilot
  bash tools/workflow.sh pack
Read docs/START_HERE.md before expanding the dataset or launching OCI jobs.
HELP
    exit 0
fi
if [[ ! -x "${NOVA_PYTHON}" ]]; then
    echo 'Run bash tools/workflow.sh setup cpu first.' >&2
    exit 1
fi
case "${NOVA_COMMAND}" in
    doctor|status|collect|synthetic|clean|mix)
        exec "${NOVA_PYTHON}" -m tabula_nova.workspace "${NOVA_COMMAND}" "$@" ;;
    test)
        exec "${NOVA_PYTHON}" tools/run_tests.py ;;
    prepare)
        exec "${NOVA_PYTHON}" -m tabula_nova prepare-text \
            --input data/mixed/pilot-v1/documents.jsonl --output prepared/workflow-smoke \
            --vocab-size 512 --tokenizer-sample-docs 512 "$@" ;;
    smoke)
        exec "${NOVA_PYTHON}" -m tabula_nova train --preset tiny --device cpu \
            --data prepared/workflow-smoke --output runs/workflow-smoke \
            --train-config configs/train-smoke.json "$@" ;;
    prepare-pilot)
        exec "${NOVA_PYTHON}" -m tabula_nova prepare-text \
            --input data/mixed/pilot-v1/documents.jsonl --output prepared/workflow-pilot \
            --vocab-size 8192 --tokenizer-sample-docs 2000 "$@" ;;
    pack)
        exec "${NOVA_PYTHON}" -m tabula_nova pack-job \
            --output "dist/job-$(date +%Y%m%d-%H%M%S)/tabula_nova_8b.zip" "$@" ;;
    *) echo "Unknown stage: ${NOVA_COMMAND}; use help." >&2; exit 2 ;;
esac
