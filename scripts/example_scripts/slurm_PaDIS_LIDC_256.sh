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
##SBATCH -A FERGUSSON-SL3-GPU
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
#SBATCH --time=0:10:00
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
export MAMBA_ROOT_PREFIX
export PATH="$MAMBA_ROOT_PREFIX/bin:$PATH"

if [ -x "$MAMBA_ROOT_PREFIX/bin/mamba" ]; then
        eval "$("$MAMBA_ROOT_PREFIX/bin/mamba" shell hook --shell bash)"
        mamba activate "$MAMBA_ROOT_PREFIX/envs/lion" || {
                echo "Could not activate the conda/mamba environment at $MAMBA_ROOT_PREFIX/envs/lion."
                exit 1
        }
        echo "Activated lion environment using mamba."
elif [ -f "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh" ]; then
        . "$MAMBA_ROOT_PREFIX/etc/profile.d/conda.sh"
        conda activate "$MAMBA_ROOT_PREFIX/envs/lion" || {
                echo "Could not activate the conda environment at $MAMBA_ROOT_PREFIX/envs/lion."
                exit 1
        }
        echo "Activated lion environment using conda."
else
        echo "Could not find mamba/conda under $MAMBA_ROOT_PREFIX."
        exit 1
fi

python - <<'PY'
try:
    import matplotlib
except ImportError:
    raise SystemExit(
        "The active 'lion' environment is missing matplotlib. "
        "Install/update it before resubmitting, for example: "
        "conda env update -f env.yml --prune"
    )
PY

#! Full path to application executable:
application="python"

#! Run options for the application:
options=(
        "$SLURM_SUBMIT_DIR/scripts/example_scripts/PaDIS_LIDC_256.py"
        --run-name padis_lidc_256_reproduction_CSD3
        --wandb-entity tjh200-university-of-cambridge
        --wandb-project PaDIS-Reproduction
        --wandb-name padis_lidc_256_reproduction_CSD3
        --wandb-mode online
        --device cuda
        --target-patches 200000000
        --validation-interval-patches 50000
        --checkpoint-interval-patches 250000
        --log-interval-patches 1000
        --batch-size 128
        --num-workers 16
        --prefetch-factor 4
)

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

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
