#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/examples/embodiment/config/libero_spatial_eval_dreamzero.yaml"
MODE="${1:-quick}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"

export DREAMZERO_SKIP_TORCH_COMPILE="${DREAMZERO_SKIP_TORCH_COMPILE:-true}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

read_config_value() {
    local key="$1"
    "${PYTHON_BIN}" - "$CONFIG_PATH" "$key" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

config_path = Path(sys.argv[1])
key = sys.argv[2]
cfg = OmegaConf.load(config_path)

mapping = {
    "model_path": cfg.actor.model.model_path,
    "tokenizer_path": cfg.actor.model.tokenizer_path,
    "embodiment_tag": cfg.actor.model.embodiment_tag,
}
value = mapping.get(key)
if value is not None:
    print(value)
PY
}

export DREAMZERO_FULL_MODEL_PATH="${DREAMZERO_FULL_MODEL_PATH:-$(read_config_value model_path)}"
export DREAMZERO_FULL_TOKENIZER_PATH="${DREAMZERO_FULL_TOKENIZER_PATH:-$(read_config_value tokenizer_path)}"
export DREAMZERO_FULL_EMBODIMENT_TAG="${DREAMZERO_FULL_EMBODIMENT_TAG:-$(read_config_value embodiment_tag)}"

run_quick() {
    echo "[DreamZero][quick] tests/unit_tests/test_dreamzero_action_head_refactor.py"
    "${PYTEST_BIN}" -q "${ROOT_DIR}/tests/unit_tests/test_dreamzero_action_head_refactor.py"
}

run_full_eval() {
    echo "[DreamZero][full-eval] tests/unit_tests/test_dreamzero_full_model_alignment.py"
    "${PYTEST_BIN}" -s -q "${ROOT_DIR}/tests/unit_tests/test_dreamzero_full_model_alignment.py"
}

run_full_forward() {
    echo "[DreamZero][full-forward] tests/unit_tests/test_dreamzero_full_model_forward_alignment.py"
    "${PYTEST_BIN}" -s -q "${ROOT_DIR}/tests/unit_tests/test_dreamzero_full_model_forward_alignment.py"
}

case "${MODE}" in
    quick)
        run_quick
        ;;
    eval)
        run_full_eval
        ;;
    forward)
        run_full_forward
        ;;
    full)
        run_quick
        run_full_eval
        run_full_forward
        ;;
    *)
        cat <<EOF
Usage: $(basename "$0") [quick|eval|forward|full]

quick    Run the lightweight action-head alignment tests.
eval     Run the full-model causal inference alignment test.
forward  Run the full-model training forward alignment test.
full     Run quick + eval + forward in order.

Defaults are read from:
  ${CONFIG_PATH}

Override with environment variables if needed:
  DREAMZERO_FULL_MODEL_PATH
  DREAMZERO_FULL_TOKENIZER_PATH
  DREAMZERO_FULL_EMBODIMENT_TAG
  PYTHON_BIN
  PYTEST_BIN
EOF
        exit 2
        ;;
esac
