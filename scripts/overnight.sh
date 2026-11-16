#!/bin/bash
# Run the WHOLE pipeline (all models, controls, Aim 2, importance, DE, ...) sequentially, unattended.
#
#   caffeinate -ims bash scripts/overnight.sh            # keeps the Mac awake; plug in the charger, leave the lid open
#   nohup bash scripts/overnight.sh > logs/overnight.out 2>&1 &     # survives closing the terminal
#
# Options (environment variables):
#   SKIP="scaling shap vs_control"   steps/groups to skip (names are printed in the plan; see STEPS below)
#                                    default: SKIP="legacy_nn" (the round-1 large-net recipe; see section 7)
#   ONLY="r2_pb r2_nn"               run only these steps/groups
#
# Groups, in order: prep, r2_pb, r2_nn, r2_scvi, r2_lbd, r2_vs_control, r2_external, r2_scaling (round 2, new),
#                   then core, controls, aim2, biology, lbd, big, vs_control, extras (round 1: finished folds are cached,
#                   so these only train what is still missing: mainly the slow cell-level models)
#   PSYCHAD_PROFILE=dev              use the fast dev profile (needs `psychad --profile dev build` first)
#   STEP_MAX_HOURS=6                 watchdog: kill one step after this long (it resumes on re-run). The slowest
#                                    steps (XGBoost, RF, scaling) have tighter caps of 3-4 h set per step below.
#   PSYCHAD_CONFIG=my.yaml           config override (default: configs/overnight.yaml = RF with 100 trees)
#   PSYCHAD_PY=/path/to/python       interpreter (default: `python`, or the psychad conda env if found)
#   RERUN_DONE=1                     recompute steps whose output file already exists (permutation null, transfer,
#                                    importance, DE, leaky split, scVI embedding); by default they are skipped
#
# Design:
#  * One step at a time. A failing step is retried once and then SKIPPED (the night continues); the summary at the end
#    lists failures. Everything is cached per (model, seed, fold), so re-running this script resumes where it stopped.
#  * Steps are ordered most-valuable and cheapest first, long optional sweeps last. If you wake up early, the
#    important results already exist.
#  * Logs: logs/overnight_<timestamp>/<step>.log, status table in status.tsv. Live view:  tail -f logs/overnight_*/status.tsv
#  * When finished it runs scripts/analyze.sh (figures + RESULTS_SUMMARY.md + sanity checks).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# keep the Mac awake for the lifetime of this script
if [ -z "${CAFFEINATED:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  CAFFEINATED=1 exec caffeinate -ims bash "$0" "$@"
fi

# ----------------------------------------------------------------------------- interpreter + preflight
PY="${PSYCHAD_PY:-python}"
if ! "$PY" -c "import psychad_ml" >/dev/null 2>&1; then
  for cand in /opt/anaconda3/envs/psychad/bin/python "$HOME/miniconda3/envs/psychad/bin/python" "$HOME/anaconda3/envs/psychad/bin/python"; do
    [ -x "$cand" ] && "$cand" -c "import psychad_ml" >/dev/null 2>&1 && PY="$cand" && break
  done
fi
PROFILE="${PSYCHAD_PROFILE:-default}"
# compute-only overrides (RF with 100 trees); set PSYCHAD_CONFIG=configs/default.yaml to run the exact defaults
PSYCHAD_CONFIG="${PSYCHAD_CONFIG:-configs/overnight.yaml}"
CFGARG="--config $PSYCHAD_CONFIG"
P="$PY -m psychad_ml --profile $PROFILE $CFGARG"
STEP_MAX_SECS=$(( ${STEP_MAX_HOURS:-6} * 3600 ))
export PYTORCH_ENABLE_MPS_FALLBACK=1        # fall back to CPU for any op MPS lacks instead of crashing
export TOKENIZERS_PARALLELISM=false

STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/overnight_$STAMP"
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/status.tsv"
printf "step\tstatus\tseconds\tattempts\tlog\n" > "$STATUS"
say() { echo "[$(date '+%a %H:%M:%S')] $*" | tee -a "$LOGDIR/overnight.log"; }

say "python: $PY   profile: $PROFILE   logs: $LOGDIR"
"$PY" - <<'EOF' || { echo "PREFLIGHT FAILED: fix the import error above, then re-run."; exit 1; }
import importlib, sys
bad = []
for m in ("psychad_ml", "numpy", "scipy", "pandas", "sklearn", "anndata", "torch", "xgboost", "pydeseq2", "gseapy", "matplotlib", "scvi"):
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001
        bad.append(f"{m}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
if bad:
    print("missing/broken packages:\n  " + "\n  ".join(bad)); sys.exit(1)
import torch
print("imports ok; torch", torch.__version__, "| mps", torch.backends.mps.is_available(), "| cuda", torch.cuda.is_available())
EOF
RUN="$("$PY" -c "from psychad_ml.config import load_config, get_paths; c=load_config('$PROFILE' if '$PROFILE'!='default' else None, '${PSYCHAD_CONFIG:-}' or None); print(c['run_name'])")"
DD="$("$PY" -c "from psychad_ml.config import load_config, get_paths; c=load_config('$PROFILE' if '$PROFILE'!='default' else None, '${PSYCHAD_CONFIG:-}' or None); print(get_paths(c).derived)" 2>/dev/null || echo "data/derived/$RUN")"
for f in donors.csv gene_panel.csv pseudobulk.h5ad cells_EN.h5ad cells_IN.h5ad cells_Astro.h5ad cells_Micro.h5ad cells_Oligo.h5ad cells_OPC.h5ad; do
  [ -f "$DD/$f" ] || { echo "PREFLIGHT FAILED: missing $DD/$f (run 'psychad build' first)"; exit 1; }
done
RESDIR="$("$PY" -c "from psychad_ml.config import load_config, get_paths; c=load_config('$PROFILE' if '$PROFILE'!='default' else None, '${PSYCHAD_CONFIG:-}' or None); print(get_paths(c).results)" 2>/dev/null || echo "results/$RUN")"
FREE_GB=$(df -g . | awk 'NR==2{print $4}')
[ "${FREE_GB:-0}" -lt 5 ] && { echo "PREFLIGHT FAILED: only ${FREE_GB} GB free disk"; exit 1; }
if command -v pmset >/dev/null 2>&1 && pmset -g batt | grep -q "Battery Power"; then
  say "WARNING: running on battery. Plug in the charger or the Mac may sleep/throttle overnight."
fi
say "preflight ok (data in $DD, ${FREE_GB} GB free)"

# ----------------------------------------------------------------------------- step runner
FAILED=()
in_list() { local x="$1"; shift; for i in $@; do [ "$i" = "$x" ] && return 0; done; return 1; }
wanted() {   # wanted <step> <group>
  if [ -n "${ONLY:-}" ]; then in_list "$1" "$ONLY" || in_list "$2" "$ONLY" || return 1; fi
  in_list "$1" "${SKIP:-}" && return 1
  in_list "$2" "${SKIP:-}" && return 1
  return 0
}
run_once() {  # run_once <logfile> <cmd...>   (with watchdog)
  local log="$1"; shift
  "$@" >>"$log" 2>&1 &
  local pid=$!
  local cap="${MAXSECS:-$STEP_MAX_SECS}"
  ( sleep "$cap"; echo "[watchdog] step exceeded ${cap}s, killing (finished folds are cached; re-run to resume)" >>"$log"; kill -TERM "$pid" 2>/dev/null; sleep 30; kill -KILL "$pid" 2>/dev/null ) &
  local wd=$!
  wait "$pid"; local rc=$?
  kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null
  return $rc
}
step() {      # step <group> <name> <cmd...>     (prefix DONE_IF=<file>: skip when that output already exists)
  local group="$1" name="$2"; shift 2
  if ! wanted "$name" "$group"; then say "skip   $name"; printf "%s\tskipped\t0\t0\t-\n" "$name" >> "$STATUS"; return 0; fi
  if [ -n "${DONE_IF:-}" ] && [ -e "$DONE_IF" ] && [ -z "${RERUN_DONE:-}" ]; then
    say "done   $name (output exists: $DONE_IF; RERUN_DONE=1 to recompute)"; printf "%s\tcached\t0\t0\t-\n" "$name" >> "$STATUS"; return 0
  fi
  local log="$LOGDIR/$name.log" t0=$SECONDS rc=1 attempts=0
  say "START  $name   ($*)"
  for attempts in 1 2; do
    [ $rc -eq 143 ] || [ $rc -eq 137 ] && { say "step was killed by the watchdog; not retrying"; break; }
    run_once "$log" "$@" && { rc=0; break; }
    rc=$?; say "FAILED $name attempt $attempts (rc=$rc); see $log"
    [ "$attempts" = 1 ] && [ $rc -ne 143 ] && [ $rc -ne 137 ] && say "retrying $name once (cached folds are reused)"
  done
  local dt=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then say "DONE   $name in $((dt/60)) min"; printf "%s\tok\t%s\t%s\t%s\n" "$name" "$dt" "$attempts" "$log" >> "$STATUS"
  else say "GIVING UP on $name after $((dt/60)) min; continuing"; FAILED+=("$name"); printf "%s\tFAILED\t%s\t%s\t%s\n" "$name" "$dt" "$attempts" "$log" >> "$STATUS"; fi
  return 0
}

SKIP="${SKIP-legacy_nn}"

# ----------------------------------------------------------------------------- STEPS (ordered; group = first arg)
# Measured on an M4 / 16 GB, ONE fold (~287k training cells): pb_lr ~1 min, pb_class_lr ~0.6, pb_class_xgb ~1.2,
# cell_lr 6, cell_mlp 1.1, mil_attn 1.3, cell_mlp_large 1.9, cell_xgb 17.5, cell_rf >25 min. A full benchmark is
# 15 folds (5 folds x 3 seeds). The slow models (cell_xgb, cell_rf) therefore run on seed 0 only (still 5-fold
# donor-level CV); everything else uses all seeds. XGBoost and torch models are always separate processes.
ALL_TASKS="four_class lbd_group ad_vs_control pd_vs_control dlb_vs_control"
FAST="pb_lr pb_class_lr pb_class_xgb"
PERM_REPS="${PERM_REPS:-200}"

# 1. prep ---------------------------------------------------------------------------------------------------
step prep       eda               $P eda
step prep       splits            $P splits --task $ALL_TASKS
step prep       baselines         $P controls --task four_class --which covariate

# R2. Round 2 (16 Nov): improved models. See configs/default.yaml `round2_models` and README "Round 2".
# Estimated on the M4 from round-1 timings (not yet measured): r2_pb ~1.5 h per task (inner-CV decision rule = ~4x the
# fits; pb_class_enet is the slowest), r2_nn ~1 h, embed ~0.5-1 h, r2_scvi ~0.5 h. Every fold is cached: re-running resumes.
R2_PB="$("$PY" -c "from psychad_ml.config import load_config; c=load_config('$PROFILE' if '$PROFILE'!='default' else None, '${PSYCHAD_CONFIG:-}' or None); print(' '.join(c['round2_models']['pb']))")"
R2_NN="$("$PY" -c "from psychad_ml.config import load_config; c=load_config('$PROFILE' if '$PROFILE'!='default' else None, '${PSYCHAD_CONFIG:-}' or None); print(' '.join(c['round2_models']['nn']))")"
R2_SCVI="$("$PY" -c "from psychad_ml.config import load_config; c=load_config('$PROFILE' if '$PROFILE'!='default' else None, '${PSYCHAD_CONFIG:-}' or None); print(' '.join(c['round2_models']['scvi']))")"
for t in lbd_group ad_vs_control pd_vs_control dlb_vs_control; do
  step prep     "baselines_$t"   $P controls --task $t --which covariate        # seconds each; lbd_group had none
done
DONE_IF="$DD/composition.csv" step prep       composition       $P build --stages composition                  # metadata only; cached if present
step r2_pb      r2_pb4            $P benchmark --task four_class --models $R2_PB
step r2_nn      r2_nn4            $P benchmark --task four_class --models $R2_NN
DONE_IF="$DD/scvi/embed_meta.json" MAXSECS=7200 step r2_scvi   embed_scvi        $P embed
step r2_scvi    r2_scvi4          $P benchmark --task four_class --models $R2_SCVI
step r2_lbd     r2_lbd_pb         $P benchmark --task lbd_group --models $R2_PB
step r2_lbd     r2_lbd_nn         $P benchmark --task lbd_group --models $R2_NN $R2_SCVI
for t in ad_vs_control pd_vs_control dlb_vs_control; do
  step r2_vs_control "r2_$t"      $P benchmark --task $t --models $FAST pb_class_lr_filt pb_comp_lr pb_class_comp_lr $R2_NN
done
step r2_external  cohort_transfer $P cohort-transfer --task ad_vs_control
MAXSECS=10800 step r2_scaling  scaling_mlp_small  $P scaling --task four_class --model mlp_small
MAXSECS=7200  step r2_scaling  scaling_mil_small  $P scaling --task four_class --model mil_small

# 2. Aim 1, four-class: the core table -----------------------------------------------------------------------
# Order reflects what looked promising on the smoke run + real fold 0: pseudobulk models first, then cell_xgb,
# then cell_lr (seed 0: moderate evidence, mostly needed as the cell-level reference), then the neural nets (cheap;
# they were at chance in both checks, so run them for completeness and read their best_epoch/best_val_loss).
step core       bench4_fast       $P benchmark --task four_class --models $FAST
MAXSECS=12600 step core       bench4_xgb        $P benchmark --task four_class --models cell_xgb --seeds 0
step core       bench4_cell_lr    $P benchmark --task four_class --models cell_lr --seeds 0
step core       bench4_mlp        $P benchmark --task four_class --models cell_mlp
step core       bench4_mil        $P benchmark --task four_class --models mil_attn

# 3. controls that make the numbers believable --------------------------------------------------------------
DONE_IF="$RESDIR/controls/four_class/permutation_pb_lr.csv" step controls   ctl_permutation   $P controls --task four_class --which permutation --reps $PERM_REPS

# 4. Aim 2 (cell-type specificity), transfer, importance, DE ------------------------------------------------
step aim2       aim2_lr           $P celltype --task four_class --models cell_lr
DONE_IF="$RESDIR/aim2/transfer/transfer_auroc.csv" step aim2       transfer          $P transfer
DONE_IF="$RESDIR/importance/four_class/lr_importance.csv" step biology    importance        $P importance --task four_class --seeds 0
DONE_IF="$RESDIR/de/four_class/de_vs_model_overlap.csv" step biology    de                bash -c "$P de || $P de --no-enrichment"

# 5. leaky-split control (seed 0 only) and cell-type MLP --------------------------------------------
DONE_IF="$RESDIR/controls/four_class/leaky_vs_donor_split.csv" step controls   ctl_leaky         $P controls --task four_class --which leaky --seeds 0
step aim2       aim2_mlp          $P celltype --task four_class --models cell_mlp --no-curve

# 6. second task: PD+DLB merged (fallback if 4-class PD/DLB are noise) ---------------------------------------
step lbd        lbd_fast          $P benchmark --task lbd_group --models $FAST
step lbd        lbd_cell_lr       $P benchmark --task lbd_group --models cell_lr --seeds 0
step lbd        lbd_mlp           $P benchmark --task lbd_group --models cell_mlp
step lbd        lbd_mil           $P benchmark --task lbd_group --models mil_attn

# 7. big models ----------------------------------------------------------------------------------------------
# cell_mlp_large uses the round-1 training recipe that stopped at epoch 1 in every fold (2.7M+ params memorise
# donors); it is skipped by default (group legacy_nn). Run it with SKIP="" if you want it for completeness.
step legacy_nn  bench4_mlp_large  $P benchmark --task four_class --models cell_mlp_large
MAXSECS=14400 step big        bench4_rf         $P benchmark --task four_class --models cell_rf --seeds 0

# 8. one-vs-control tasks (cheap models) ----------------------------------------------------------------------
for t in ad_vs_control pd_vs_control dlb_vs_control; do
  step vs_control  "bench_$t"     $P benchmark --task $t --models $FAST
done

# 9. optional extras, very long (skipped first if you are short on time: SKIP="extras") -----------------------
MAXSECS=10800 step extras     lbd_xgb           $P benchmark --task lbd_group --models cell_xgb --seeds 0
MAXSECS=10800 step extras     lbd_rf            $P benchmark --task lbd_group --models cell_rf --seeds 0
step extras     shap              $P importance --task four_class --shap --seeds 0
MAXSECS=10800 step legacy_nn  scaling_mlp       $P scaling --task four_class --model cell_mlp
MAXSECS=10800 step legacy_nn  scaling_mil       $P scaling --task four_class --model mil_attn --sweeps cells genes

# ----------------------------------------------------------------------------- wrap up
say "all steps attempted; running analysis (figures, RESULTS_SUMMARY.md, sanity checks)"
PSYCHAD_PY="$PY" PSYCHAD_PROFILE="$PROFILE" PSYCHAD_CONFIG="${PSYCHAD_CONFIG:-}" bash scripts/analyze.sh > "$LOGDIR/analyze.log" 2>&1 \
  && say "analysis ok -> $RESDIR/RESULTS_SUMMARY.md" || say "analysis had problems; see $LOGDIR/analyze.log"
echo; echo "=========== OVERNIGHT SUMMARY ($LOGDIR) ==========="
column -t -s $'\t' "$STATUS" 2>/dev/null || cat "$STATUS"
if [ ${#FAILED[@]} -gt 0 ]; then echo; echo "FAILED steps: ${FAILED[*]}   (read $LOGDIR/<step>.log; fix; re-run this script: finished work is cached)"; else echo; echo "All steps succeeded."; fi
echo "Next: open $RESDIR/RESULTS_SUMMARY.md and $RESDIR/figures/   (details: $LOGDIR/analyze.log)"
