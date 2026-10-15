#!/usr/bin/env bash
set -euo pipefail

NOVA_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${NOVA_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

: "${NOVA_DATA_DIR:?Set NOVA_DATA_DIR to a local prepared corpus or shared filesystem path}"
: "${NOVA_OUTPUT_DIR:?Set NOVA_OUTPUT_DIR to persistent checkpoint storage}"
NOVA_GPUS_PER_NODE="${NOVA_GPUS_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
NOVA_NNODES="${NOVA_NNODES:-1}"
NOVA_NODE_RANK="${NOVA_NODE_RANK:-0}"
NOVA_MASTER_PORT="${NOVA_MASTER_PORT:-29500}"
if (( NOVA_GPUS_PER_NODE < 1 )); then
    echo 'A GPU shape is required for the 8B job.' >&2
    exit 1
fi

python -m tabula_nova inspect --preset 8b --verify-meta --world-size "$((NOVA_GPUS_PER_NODE * NOVA_NNODES))"
NOVA_TORCHRUN_ARGS=(--nproc_per_node "${NOVA_GPUS_PER_NODE}" --nnodes "${NOVA_NNODES}")
if (( NOVA_NNODES == 1 )); then
    NOVA_TORCHRUN_ARGS+=(--standalone)
else
    : "${NOVA_MASTER_ADDR:?Set NOVA_MASTER_ADDR to rank-zero private IP for multinode jobs}"
    NOVA_TORCHRUN_ARGS+=(--node_rank "${NOVA_NODE_RANK}" --master_addr "${NOVA_MASTER_ADDR}" --master_port "${NOVA_MASTER_PORT}")
fi
NOVA_TRAIN_ARGS=(--preset "${NOVA_PRESET:-8b}" --device cuda --fsdp
    --data "${NOVA_DATA_DIR}" --output "${NOVA_OUTPUT_DIR}"
    --train-config "${NOVA_TRAIN_CONFIG:-${NOVA_PROJECT_ROOT}/configs/train-oci.json}")
if [[ -n "${NOVA_RESUME:-}" ]]; then
    NOVA_TRAIN_ARGS+=(--resume "${NOVA_RESUME}")
fi
python -m torch.distributed.run "${NOVA_TORCHRUN_ARGS[@]}" -m tabula_nova train "${NOVA_TRAIN_ARGS[@]}" "$@"

# Every rank writes shards into the same mounted output path. Only node zero
# transfers completed checkpoints after all local workers have finished.
if [[ -n "${NOVA_OUTPUT_URI:-}" ]] && (( NOVA_NODE_RANK == 0 )); then
    python -m tabula_nova upload --path "${NOVA_OUTPUT_DIR}" --uri "${NOVA_OUTPUT_URI}" --auth "${NOVA_OCI_AUTH:-resource_principal}"
fi
