#!/bin/bash
# Run on your laptop. Nothing is computed here: it copies files and talks to the cluster over ssh.
#   bash hpc/hpc.sh go        upload + setup + submit
#   bash hpc/hpc.sh status    queue and progress
#   bash hpc/hpc.sh fetch     copy the results back into hpc/results/
# Single steps: upload | setup | test | submit | report | here      Settings: hpc/config.sh
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
source hpc/config.sh
SSH=(ssh -o ControlMaster=auto -o ControlPath="$HOME/.ssh/vh-%C" -o ControlPersist=10m)

remote() { "${SSH[@]}" "$HPC_HOST" "bash -lc 'cd \"$HPC_DIR\" && bash hpc/job.sh $*'"; }

upload() {
    "${SSH[@]}" "$HPC_HOST" "mkdir -p \"$HPC_DIR\""
    rsync -azR -e "${SSH[*]}" --exclude __pycache__ --exclude venv --exclude logs --exclude 'results*' \
        hpc claude-preprocessing/prep.py train-template-kit/train.csv "$HPC_HOST:$HPC_DIR/"
    echo "uploaded to $HPC_HOST:$HPC_DIR"
}

case "${1:-}" in
    go)      upload; remote setup; remote submit ;;
    upload)  upload ;;
    setup | test | status | here) remote "$1" ;;
    submit)  remote submit ;;
    report)  remote submit report ;;
    fetch)
        mkdir -p hpc/results hpc/logs
        rsync -az -e "${SSH[*]}" "$HPC_HOST:$HPC_DIR/hpc/results/" hpc/results/
        rsync -az -e "${SSH[*]}" "$HPC_HOST:$HPC_DIR/hpc/logs/" hpc/logs/ || true
        echo "$(ls hpc/results/runs 2>/dev/null | grep -c 'json$') model settings fetched"
        if [[ -f hpc/results/best.json ]]; then
            echo "settings to use: hpc/results/best.json, explanation: hpc/results/RESULTS.md"
        else
            echo "no best.json yet: the report job has not finished (bash hpc/hpc.sh status)"
        fi
        ;;
    *)
        sed -n '2,6p' "$0"
        exit 1
        ;;
esac
