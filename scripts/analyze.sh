#!/bin/bash
# Post-training analysis: figures + RESULTS_SUMMARY.md + sanity checks. Safe on partial results; re-run any time.
#
#   bash scripts/analyze.sh            # everything below
#   bash scripts/analyze.sh --open     # also open the figures folder and the summary (macOS)
#
# Writes (under results/<run>/):
#   figures/*.png         EDA, model comparison (macro-F1, balanced accuracy, AUROC), recall by class, paired difference
#                         vs baseline, confusion matrices, leaky-vs-donor, permutation null, per-cell-class,
#                         downsampling, transfer matrix (with CIs), scaling
#   RESULTS_SUMMARY.md    all tables + verdicts vs the covariate/prior baselines + what finished / what is missing
# and prints PASS/WARN/FAIL sanity checks of the saved predictions (scripts/check_results.py).
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PSYCHAD_PY:-python}"
if ! "$PY" -c "import psychad_ml" >/dev/null 2>&1 && [ -x /opt/anaconda3/envs/psychad/bin/python ]; then PY=/opt/anaconda3/envs/psychad/bin/python; fi
PROFILE="${PSYCHAD_PROFILE:-default}"
CFG="${PSYCHAD_CONFIG:-}"
P="$PY -m psychad_ml --profile $PROFILE ${CFG:+--config $CFG}"
PROF_ARG="$([ "$PROFILE" != default ] && echo "$PROFILE" || echo None)"
RES="$("$PY" -c "
from psychad_ml.config import load_config, get_paths
p = '$PROFILE'
c = load_config(None if p == 'default' else p, '$CFG' or None)
print(get_paths(c).results)")"

# tasks that actually have results
TASKS=""
for d in "$RES"/aim1/*/; do [ -d "$d" ] && TASKS="$TASKS $(basename "$d")"; done
[ -z "$TASKS" ] && { echo "No results in $RES/aim1 yet: nothing to analyse. Train first (scripts/overnight.sh)."; exit 1; }
echo "tasks with results:$TASKS"

echo; echo "== 1/3 summary tables, paired tests vs baseline, RESULTS_SUMMARY.md (reads saved predictions only; trains nothing) =="
$P summary --task $TASKS > "$RES/summary_stdout.txt" 2>&1 && echo "wrote $RES/RESULTS_SUMMARY.md (aim1/<task>/summary.csv + paired_vs_baseline.csv refreshed)" || echo "summary failed; see $RES/summary_stdout.txt"

echo; echo "== 2/3 figures =="
$P report --task $TASKS 2>&1 | grep -v -i "warn" | tail -60

echo; echo "== 3/3 sanity checks =="
"$PY" scripts/check_results.py --profile "$PROFILE" ${CFG:+--config $CFG} 2>&1 | grep -v -i "warn(" | tail -60

echo; echo "Read first:  $RES/RESULTS_SUMMARY.md     figures: $RES/figures/"
if [ "${1:-}" = "--open" ] && command -v open >/dev/null 2>&1; then open "$RES/figures" "$RES/RESULTS_SUMMARY.md"; fi
