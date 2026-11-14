#!/bin/bash
# Runs on the cluster, inside the uploaded folder. hpc/hpc.sh calls these steps for you.
#   bash hpc/job.sh setup            create hpc/venv (search) and hpc/venv-submission (pinned, submission/requirements.txt)
#   bash hpc/job.sh test             tiny end-to-end run on one core (about 3 minutes)
#   bash hpc/job.sh submit           queue the search (array job) and the report (runs after it)
#   bash hpc/job.sh submit report    queue only the report
#   bash hpc/job.sh status           queue and progress
#   bash hpc/job.sh check            queue the end-to-end check of submission/ (log: hpc/logs/submission.out)
#   bash hpc/job.sh export           queue training + saving submission/model.joblib (log: hpc/logs/model.out)
#   bash hpc/job.sh here             no scheduler: run all tasks on this machine
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
source hpc/config.sh
PY=${VENV:-$PWD/hpc/venv}/bin/python
SUB_PY=$PWD/hpc/venv-submission/bin/python
STEP=${1:?usage: bash hpc/job.sh setup|test|submit|status|here}

modules() {
    if [[ -n "$MODULES" ]]; then
        set +eu
        type module >/dev/null 2>&1 || source /etc/profile
        module load $MODULES
        set -eu
    fi
}

queue() {  # queue <step> <minutes> [array size] [job id to wait for] -> prints the new job id
    local step=$1 minutes=$2 tasks=${3:-} after=${4:-} options
    if command -v sbatch >/dev/null 2>&1; then
        options=(--parsable --job-name="vh_$step" --time="$minutes" --cpus-per-task="$CPUS"
                 --mem="$((CPUS * MEM_GB_PER_CORE))G")
        if [[ -n "$QUEUE" ]]; then options+=(--partition="$QUEUE"); fi
        if [[ -n "$ACCOUNT" ]]; then options+=(--account="$ACCOUNT"); fi
        if [[ -n "$after" ]]; then options+=(--dependency="afterany:$after"); fi
        if [[ -n "$tasks" ]]; then
            options+=(--array="0-$((tasks - 1))" --output="hpc/logs/${step}_%a.out")
        else
            options+=(--output="hpc/logs/$step.out")
        fi
        sbatch "${options[@]}" --wrap="bash hpc/job.sh $step" | cut -d';' -f1
    elif command -v bsub >/dev/null 2>&1; then
        options=(-n "$CPUS" -R "span[hosts=1]" -R "rusage[mem=${MEM_GB_PER_CORE}GB]"
                 -M "$((MEM_GB_PER_CORE + 1))GB" -W "$minutes")
        if [[ -n "$QUEUE" ]]; then options+=(-q "$QUEUE"); fi
        if [[ -n "$after" ]]; then options+=(-w "ended($after)"); fi
        if [[ -n "$tasks" ]]; then
            options+=(-J "vh_$step[1-$tasks]" -o "hpc/logs/${step}_%I.out")
        else
            options+=(-J "vh_$step" -o "hpc/logs/$step.out")
        fi
        bsub "${options[@]}" "bash hpc/job.sh $step" | grep -o '<[0-9]*>' | head -1 | tr -d '<>'
    else
        echo "no sbatch or bsub on this machine: use 'bash hpc/job.sh here' on a compute node" >&2
        return 1
    fi
}

case "$STEP" in
    setup)
        modules
        if ! python3 -c "import sys; sys.exit(sys.version_info < (3, 9))"; then
            echo "$(python3 --version) is too old: set MODULES in hpc/config.sh (see: module avail python3)" >&2
            exit 1
        fi
        [[ -x "$PY" ]] || python3 -m venv "$(dirname "$(dirname "$PY")")"
        "$PY" -m pip install --quiet --upgrade pip
        "$PY" -m pip install --quiet -r hpc/requirements.txt
        "$PY" -c "import sys, pandas, sklearn, lightgbm; print('python', sys.version.split()[0], '| pandas', pandas.__version__, '| scikit-learn', sklearn.__version__, '| lightgbm', lightgbm.__version__)"
        echo "model settings to search: $("$PY" hpc/search.py --count)"
        [[ -x "$SUB_PY" ]] || python3 -m venv "$(dirname "$(dirname "$SUB_PY")")"
        "$SUB_PY" -m pip install --quiet --upgrade pip
        "$SUB_PY" -m pip install --quiet -r submission/requirements.txt
        "$SUB_PY" -c "import sys, numpy, pandas, sklearn, lightgbm; print('submission: python', sys.version.split()[0], '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| scikit-learn', sklearn.__version__, '| lightgbm', lightgbm.__version__)"
        ;;
    test)
        modules
        export VH_SMOKE=1 VH_RESULTS=$PWD/hpc/results_test OMP_NUM_THREADS=1
        rm -rf "$VH_RESULTS"
        "$PY" hpc/search.py --task 0 --tasks 1
        "$PY" hpc/search.py --references
        "$PY" hpc/report.py > /dev/null
        "$PY" hpc/tuned_solution.py --check
        rm -rf "$VH_RESULTS"
        echo "test passed"
        ;;
    submit)
        mkdir -p hpc/logs
        if [[ "${2:-}" == check ]]; then
            echo "submission check job $(queue submission "$REPORT_MIN") (at most $REPORT_MIN min, log: hpc/logs/submission.out)"
        elif [[ "${2:-}" == export ]]; then
            echo "model export job $(queue model "$REPORT_MIN") (at most $REPORT_MIN min, log: hpc/logs/model.out)"
        elif [[ "${2:-}" == report ]]; then
            echo "report job $(queue report "$REPORT_MIN")"
        else
            search=$(queue search "$SEARCH_MIN" "$TASKS")
            report=$(queue report "$REPORT_MIN" "" "$search")
            echo "search job $search ($TASKS tasks, at most $SEARCH_MIN min), report job $report (at most $REPORT_MIN min, starts when the search has ended)"
        fi
        ;;
    search)
        modules
        export OMP_NUM_THREADS=$CPUS
        task=${SLURM_ARRAY_TASK_ID:-$((${LSB_JOBINDEX:-1} - 1))}
        "$PY" hpc/search.py --task "$task" --tasks "$TASKS" --budget-min "$BUDGET_MIN"
        ;;
    report)
        modules
        export OMP_NUM_THREADS=$CPUS
        "$PY" hpc/search.py --references
        "$PY" hpc/report.py
        "$PY" hpc/tuned_solution.py --check
        ;;
    check)
        bash hpc/job.sh submit check
        ;;
    export)
        bash hpc/job.sh submit export
        ;;
    submission | model)
        modules
        export OMP_NUM_THREADS=$CPUS
        if [[ ! -x "$SUB_PY" ]]; then
            echo "hpc/venv-submission is missing: run 'bash hpc/hpc.sh setup' first" >&2
            exit 1
        fi
        if [[ "$STEP" == submission ]]; then
            "$SUB_PY" submission/check.py
        else
            "$SUB_PY" submission/export_model.py
        fi
        ;;
    status)
        modules
        if command -v squeue >/dev/null 2>&1; then
            squeue -u "$USER" || true
        elif command -v bjobs >/dev/null 2>&1; then
            bjobs -A 2>/dev/null || true
            bjobs -J vh_report 2>/dev/null || true
        fi
        echo "finished model settings: $(ls hpc/results/runs 2>/dev/null | grep -c 'json$') of $("$PY" hpc/search.py --count)"
        if [[ -f hpc/results/best.json ]]; then
            echo "report: ready, fetch it with 'bash hpc/hpc.sh fetch'"
        else
            echo "report: not written yet"
        fi
        if [[ -f submission/model.joblib ]]; then
            echo "model.joblib: ready, fetch it with 'bash hpc/hpc.sh fetch'"
        fi
        ;;
    here)
        modules
        mkdir -p hpc/logs
        export OMP_NUM_THREADS=1
        for task in $(seq 0 $((TASKS - 1))); do
            "$PY" hpc/search.py --task "$task" --tasks "$TASKS" --budget-min "$BUDGET_MIN" > "hpc/logs/search_$task.out" 2>&1 &
        done
        wait
        bash hpc/job.sh report
        ;;
    *)
        echo "unknown step: $STEP" >&2
        exit 1
        ;;
esac
