#!/bin/bash
# Run the pipeline in a CPU-first order. Everything is resumable: re-run after an interruption.
#
#   bash scripts/run_all.sh                 # stages 0-2 (data + CPU models + controls)
#   bash scripts/run_all.sh gpu             # stage 3 (MLP / MIL; GPU recommended)
#   bash scripts/run_all.sh extra           # stage 4 (large MLP, scaling sweeps, SHAP; optional)
#   bash scripts/run_all.sh all             # everything
#   PSYCHAD_PROFILE=dev bash scripts/run_all.sh    # fast profile (10x less data)
set -euo pipefail
cd "$(dirname "$0")/.."
P="python -m psychad_ml --profile ${PSYCHAD_PROFILE:-default}"
STAGE="${1:-cpu}"

data() {
  $P inspect                      # read the printed tables before building
  $P build                        # ~2 h from Census (resumable)
  $P eda
  $P splits --task four_class lbd_group ad_vs_control pd_vs_control dlb_vs_control
}
cpu_models() {                    # stage 1+2: no GPU needed
  for t in four_class lbd_group; do
    $P benchmark --task "$t" --models cell_lr pb_lr pb_class_lr pb_class_xgb cell_rf cell_xgb
  done
  $P controls --task four_class   # covariate + prior baselines, leaky split, permutation null
  $P celltype --task four_class --models cell_lr
  $P transfer
  $P importance --task four_class
  $P de
}
gpu_models() {                    # stage 3: only worth it if stage 1-2 show signal above the baselines
  $P benchmark --task four_class lbd_group --models cell_mlp mil_attn
  $P celltype --task four_class --models cell_mlp
}
extra() {                         # stage 4: optional
  $P benchmark --task four_class --models cell_mlp_large
  $P scaling --task four_class
  $P importance --task four_class --shap
}

case "$STAGE" in
  cpu)   data; cpu_models; $P report --task four_class ;;
  gpu)   gpu_models; $P report --task four_class ;;
  extra) extra; $P report --task four_class ;;
  all)   data; cpu_models; gpu_models; extra; $P report --task four_class ;;
  *) echo "usage: $0 [cpu|gpu|extra|all]"; exit 1 ;;
esac
