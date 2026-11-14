# Settings for hpc/hpc.sh (laptop) and hpc/job.sh (cluster).

HPC_HOST="${HPC_HOST:-dtu}"   # ssh login: a Host from ~/.ssh/config, or user@login.node
HPC_DIR="valhackers"          # folder on the cluster, relative to your home directory there

TASKS=32                      # parallel array tasks
CPUS=2                        # cores per task
MEM_GB_PER_CORE=2
BUDGET_MIN=30                 # a task starts no new model setting after this many minutes
SEARCH_MIN=$((BUDGET_MIN + 8))    # hard limit per search task
REPORT_MIN=20                     # hard limit of the report job

QUEUE=""                      # LSF queue / SLURM partition, empty = the cluster's default
ACCOUNT=""                    # SLURM account, if your cluster needs one
MODULES="python3/3.12.4"                    # e.g. "python3/3.11.9" (see: module avail python3), empty = system python3
