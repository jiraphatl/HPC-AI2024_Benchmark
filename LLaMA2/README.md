# LitGPT Llama2 Fine-tuning Setup

This readme provides instructions for setting up and running the LitGPT framework to fine-tune the Llama2 language model on the Alpaca dataset. The setup is provided for two different HPC systems: GADI and ASPIRE.

## Prerequisites
- Access to the GADI or ASPIRE HPC system
- Familiarity with the command line and PBS job submission process

## GADI Setup

### Install LitGPT
1. Create a workspace directory for Llama2 in the scratch directory:
   ```bash
   mkdir ${HOME}/scratch/workdir/llama -p
   ```
2. Set up a Python 3 Conda environment:
   ```bash
   time conda create -p ${HOME}/scratch/workdir/llama/litgpt.py312 python=3.12 -y
   ```
3. Install the pre-built LitGPT from PyPI:
   ```bash
   time ${HOME}/scratch/workdir/llama/litgpt.py312/bin/pip install 'litgpt[all]'==0.4.12
   ```

### Enable MPI Support for LitGPT Pytorch
1. Load the necessary environment modules:
   ```bash
   module purge
   module load openmpi/4.1.5
   ```
2. Install `mpi4py` from the shell to enable MPI Support:
   ```bash
   time LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pbs/default/lib:/lib64 \
   MPI4PY_BUILD_MPICC=$OMPI_ROOT/bin/mpicc \
   ${HOME}/scratch/workdir/llama/litgpt.py312/bin/pip \
   install --no-cache-dir mpi4py
   ```

## ASPIRE Setup

### Install LitGPT
1. Create a workspace directory for Llama2 in the scratch directory:
   ```bash
   mkdir ${HOME}/scratch/workdir/llama -p
   ```
2. Set up a Python 3 Conda environment:
   ```bash
   time ${HOME}/miniconda/bin/conda create -p ${HOME}/scratch/workdir/llama/litgpt.py312 python=3.12 -y
   ```
3. Install the pre-built LitGPT from PyPI:
   ```bash
   time ${HOME}/scratch/workdir/llama/litgpt.py312/bin/pip install 'litgpt[all]'==0.4.12
   ```

### Enable MPI Support for LitGPT Pytorch
1. Load the necessary environment modules:
   ```bash
   module purge
   module load openmpi/4.1.2-hpe
   module load libfabric/1.11.0.4.125
   ```
2. Install `mpi4py` from the shell to enable MPI Support:
   ```bash
   time LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/pbs/lib:/opt/cray/pe/pmi/6.1.1/lib:/opt/cray/pe/lib64:/lib64 \
   MPI4PY_BUILD_MPICC=$OPENMPI_DIR/bin/mpicc \
   ${HOME}/scratch/workdir/llama/litgpt.py312/bin/pip \
   install --no-cache-dir mpi4py
   ```

## Prepare Dataset and LitGPT Configuration
1. Create directories for the Llama2 model files and Alpaca dataset:
   ```bash
   mkdir -p ${HOME}/scratch/workdir/llama/model/litgpt/meta-llama/Llama-2-7b-hf
   mkdir -p ${HOME}/scratch/workdir/llama/dataset
   ```
2. Copy the Llama2 model files and Alpaca dataset from the shared storage:
   ```bash
   time rsync -avSP /scratch/public/2024-apac-hpc-ai/llama/model/litgpt/meta-llama/Llama-2-7b-hf/ ${HOME}/scratch/workdir/llama/model/litgpt/meta-llama/Llama-2-7b-hf/
   time rsync -avSP /scratch/public/2024-apac-hpc-ai/llama/dataset/ ${HOME}/scratch/workdir/llama/dataset/
   ```
3. Create a LitGPT fine-tuning configuration file `${HOME}/scratch/workdir/llama/full.yaml`:
   ```yaml
   precision: bf16-true
   resume: false
   train:
     save_interval: 20000
     log_interval: 1
     epochs: 1
     max_steps:
     max_seq_length: 512
   eval:
     interval: 25000
     initial_validation: false
     final_validation: false
   logger_name: csv
   ```

## Run the Distributed Finetune-full Task

### GADI PBS Script
Create a script file `${HOME}/run/llama.sh` with the following contents:

```bash
#!/bin/bash
#PBS -j oe
#PBS -l walltime=00:00:200
#PBS -m abe
#PBS -M 393958790@qq.com
#PBS -P xs75
#PBS -q gpuvolta

date
module purge
module load pbs openmpi/4.1.5

env
cat $PBS_NODEFILE

cmd="mpirun \
-wdir ${HOME}/scratch/workdir/llama \
-output-filename ${HOME}/run/output/${PBS_JOBNAME}.${PBS_JOBID} \
-map-by ppr:4:node -oversubscribe \
-report-bindings \
-x NCCL_DEBUG=INFO \
-x NCCL_NET_GDR_LEVEL=6 \
${HOME}/scratch/workdir/llama/litgpt.py312/bin/litgpt \
finetune_full \
${HOME}/scratch/workdir/llama/model/litgpt/meta-llama/Llama-2-7b-hf \
--out_dir ${HOME}/scratch/workdir/llama/out/finetune/full \
--data JSON --data.json_path ${HOME}/scratch/workdir/llama/dataset/alpaca1024 \
--config ${HOME}/scratch/workdir/llama/full.yaml \
--eval.final_validation=false \
--train.epochs=1 \
--devices=4 --num_nodes=2 \
--train.max_steps=${max_steps} \
--train.global_batch_size=${global_batch_size} \
--train.micro_batch_size=${micro_batch_size}"

echo ${cmd}

exec ${cmd}
date
```

### ASPIRE PBS Script
Create a script file `${HOME}/run/llama.sh` with the following contents:

```bash
#!/bin/bash
#PBS -j oe
#PBS -l walltime=00:00:200
#PBS -m abe
#PBS -M 393958790@qq.com
#PBS -P xs75
#PBS -q gpuvolta

module purge
module load openmpi/4.1.2-hpe
module load libfabric/1.11.0.4.125

date
env
cat $PBS_NODEFILE

# While you may try enabling RDMA in the command line and get better performance, a quick way to achieve a workable distributed training is to disable it..
cmd="mpirun \
-wdir ${HOME}/scratch/workdir/llama \
-output-filename ${HOME}/run/output/${PBS_JOBNAME}.${PBS_JOBID} \
-map-by ppr:4:node -oversubscribe \
-report-bindings \
-x NCCL_DEBUG=INFO \
-x NCCL_NET_GDR_LEVEL=0 \
-x NCCL_IB_DISABLE=1 \
-mca pml ^ucx \
${HOME}/scratch/workdir/llama/litgpt.py312/bin/litgpt \
finetune_full \
${HOME}/scratch/workdir/llama/model/litgpt/meta-llama/Llama-2-7b-hf \
--out_dir ${HOME}/scratch/workdir/llama/out/finetune/full \
--data JSON --data.json_path ${HOME}/scratch/workdir/llama/dataset/alpaca1024 \
--config ${HOME}/scratch/workdir/llama/full.yaml \
--eval.final_validation=false \
--train.epochs=1 \
--devices=4 --num_nodes=2 \
--train.max_steps=${max_steps} \
--train.global_batch_size=${global_batch_size} \
--train.micro_batch_size=${micro_batch_size}"

echo ${cmd}

exec ${cmd}
date
```

### Submit the Job
To submit the job to the PBS queue, use the following command:

```bash
cd ${HOME}/run

# GADI
nodes=2 walltime=7201 \
global_batch_size=128 micro_batch_size=32 max_steps=20 \
bash -c \
'qsub -V \
-l walltime=${walltime},ncpus=$((${nodes}*4*12)),mem=$((${nodes}*4*32))gb,ngpus=$((${nodes}*4)) \
-N llama.nodes${nodes}.GBS${global_batch_size}.MBS${micro_batch_size} \
llama.sh'

# ASPIRE
nodes=2 walltime=7201 \
global_batch_size=128 micro_batch_size=32 max_steps=20 \
bash -c \
'qsub -V \
-l walltime=${walltime},select=${nodes}:ngpus=4 \
-N llama.nodes${nodes}.GBS${global_batch_size}.MBS${micro_batch_size} \
llama.sh'
```

## Check Runtime Log
To monitor the job's progress, you can follow the runtime log:

```bash
# GADI
tail -f ${HOME}/run/output/llama.nodes2.GBS64.MBS8.{PBS_JOBNAME.PBS_JOBID}.gadi-pbs/1/rank.*/std*

# ASPIRE
tail -f ${HOME}/run/output/llama.nodes2.GBS64.MBS8.{PBS_JOBNAME.PBS_JOBID}.pbs-101/1/rank.*/std*
```

## Read the Results
The performance results of the LitGPT Llama2 training are measured in "Training time". The lower the value, the better.

```bash
# GADI
grep "Training time" ${HOME}/run/output/llama.*/1/rank.*/*

# ASPIRE
grep "Training time" ${HOME}/run/output/llama.*/1/rank.*/*
```