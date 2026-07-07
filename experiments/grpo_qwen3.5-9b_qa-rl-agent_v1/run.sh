#!/usr/bin/env bash
# QA-RL 多轮检索 Agent。实验目录包含 run.py，通用脚本会自动选用它。
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${EXP_DIR}/../../scripts/_run_experiment.sh" "${EXP_DIR}"
