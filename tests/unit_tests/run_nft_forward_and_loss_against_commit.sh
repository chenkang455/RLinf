#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BASELINE_COMMIT="${1:-4498ca4866eb9d704d8170330c4d15a75657398a}"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

python tests/unit_tests/compare_nft_forward_and_loss_against_commit.py \
  --commit "${BASELINE_COMMIT}"
