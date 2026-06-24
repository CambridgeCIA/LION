#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#!
# set -euo pipefail

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J PaDIS_LIDC_256
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A MPHIL-DIS-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#! How much wallclock time will be required?
#SBATCH --time=12:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime.

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:

MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/home/tjh200/miniforge3}"
LION_MAMBA_ENV="${LION_MAMBA_ENV:-lion}"
export MAMBA_ROOT_PREFIX
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

if [ ! -x "$MAMBA_ROOT_PREFIX/bin/mamba" ]; then
        echo "Could not find mamba under $MAMBA_ROOT_PREFIX."
        exit 1
fi
eval "$("$MAMBA_ROOT_PREFIX/bin/mamba" shell hook --shell bash)"
mamba activate "$LION_MAMBA_ENV" || {
        echo "Could not activate mamba environment $LION_MAMBA_ENV."
        exit 1
}
echo "Activated $LION_MAMBA_ENV environment using mamba."

python - <<'PY'
try:
    import matplotlib
except ImportError:
    raise SystemExit(
        "The active 'lion' environment is missing matplotlib. "
        "Install/update it before resubmitting, for example: "
        "mamba env update -n lion -f env.yml --prune"
    )
PY

#! Full path to application executable:
application="python"

#! Resume settings. Edit these directly, or override them at submission time.
RUN_NAME="${PADIS_RUN_NAME:-padis_lidc_256_reproduction_CSD3}"
#! Optional: paste the WandB run id here, e.g. WANDB_ID="abc123xyz".
WANDB_ID="a6hr0vvo"
if [ -n "${PADIS_SAVE_FOLDER:-}" ]; then
        SAVE_FOLDER="$PADIS_SAVE_FOLDER"
elif [ -n "${LION_EXPERIMENTS_PATH:-}" ]; then
        SAVE_FOLDER="$LION_EXPERIMENTS_PATH/PaDIS/LIDC_256"
elif [ -n "${LION_DATA_PATH:-}" ]; then
        SAVE_FOLDER="$LION_DATA_PATH/experiments/PaDIS/LIDC_256"
else
        echo "Set LION_DATA_PATH, LION_EXPERIMENTS_PATH, or PADIS_SAVE_FOLDER so the previous PaDIS run can be found."
        exit 1
fi
RUN_FOLDER="$SAVE_FOLDER/$RUN_NAME"

if [ ! -d "$RUN_FOLDER" ]; then
        echo "Cannot resume: run folder does not exist: $RUN_FOLDER"
        echo "Set PADIS_SAVE_FOLDER and/or PADIS_RUN_NAME to the existing run."
        exit 1
fi

shopt -s nullglob
resume_files=(
        "$RUN_FOLDER"/padis_lidc_256_checkpoint_*.pt
        "$RUN_FOLDER"/padis_lidc_256_full.pt
        "$RUN_FOLDER"/padis_lidc_256_min_val_full.pt
)
shopt -u nullglob
if [ "${#resume_files[@]}" -eq 0 ]; then
        echo "Cannot resume: no PaDIS checkpoint or full training state found in $RUN_FOLDER"
        echo "Expected one of: padis_lidc_256_checkpoint_*.pt, padis_lidc_256_full.pt, padis_lidc_256_min_val_full.pt"
        exit 1
fi

#! Run options for the application:
options=(
        "$SLURM_SUBMIT_DIR/scripts/paper_scripts/PaDIS/PaDIS_LIDC_256.py"
        --save-folder "$SAVE_FOLDER"
        --run-name "$RUN_NAME"
        --wandb-entity tjh200-university-of-cambridge
        --wandb-project PaDIS-Reproduction
        --wandb-name "$RUN_NAME"
        --wandb-mode online
        --device cuda
        --target-patches 40000000
        --validation-interval-patches 10000
        --checkpoint-interval-patches 250000
        --log-interval-patches 128
        --seed "${PADIS_SEED:-33}"
        --batch-size 128
        --num-workers 16
        --prefetch-factor 4
        --cache-dataset ramdisk
        --cache-folder "/ramdisks/$USER/lion_lidc_cache"
)

if [ "${PADIS_NO_WANDB_ARTIFACT:-1}" = "1" ]; then
        options+=(--no-wandb-artifact)
fi

if [ -n "$WANDB_ID" ]; then
        options+=(--wandb-id "$WANDB_ID")
fi

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1
export PYTHONHASHSEED="${PADIS_SEED:-33}"

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD=("$application" "${options[@]}")

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd "$workdir"
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat "$NODEFILE" | uniq > "$workdir/machine.file.$JOBID"
        echo -e "\nNodes allocated:\n================"
        sed -e 's/\..*$//g' "$workdir/machine.file.$JOBID"
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n=================="
printf '%q ' "${CMD[@]}"
printf '\n\n'

"${CMD[@]}"
