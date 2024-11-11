#!/bin/bash
#PBS -j oe
#PBS -l walltime=00:00:200
#PBS -m abe
#PBS -M peera.sit@dome.tu.ac.th
#PBS -P jq90
#PBS -q gpuvolta

module purge
module load pbs openmpi/4.1.5
module load nvhpc-profilers/24.7
module load cuda/11.8
module load pytorch
# Display environment variables and allocated nodes
env
cat $PBS_NODEFILE
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set paths and configurations
HOME_DIR="${HOME}/scratch/workdir/llama"
OUTPUT_DIR="${HOME}/run/output"
MODEL_DIR="${HOME}/scratch/workdir/llama/model/litgpt/meta-llama/Llama-2-7b-hf"
OUT_DIR="${HOME}/scratch/workdir/llama/out/finetune/full"
DATA_JSON_PATH="${HOME}/scratch/workdir/llama/dataset/alpaca1024"
CONFIG_FILE="${HOME}/scratch/workdir/llama/full.yaml"
NSYS_OUTPUT="nsys_llama2_nsight_trace"


# Set up the mpirun command with fine-tuning task
cmd="mpirun \
-wdir ${HOME_DIR} \
-output-filename ${OUTPUT_DIR}/${PBS_JOBNAME}.${PBS_JOBID} \
-map-by ppr:${DEVICES_PER_NODE}:node -oversubscribe \
-report-bindings \
-x NCCL_DEBUG=INFO \
-x NCCL_NET_GDR_LEVEL=6 \
${HOME_DIR}/litgpt.py312/bin/litgpt \
finetune_full \
${MODEL_DIR} \
--out_dir ${OUT_DIR} \
--data JSON --data.json_path ${DATA_JSON_PATH} \
--config ${CONFIG_FILE} \
--eval.final_validation=false \
--train.epochs=1 \
--devices=${DEVICES_PER_NODE} --num_nodes=${NUM_NODES} \
--train.max_steps=${max_steps} \
--train.global_batch_size=${global_batch_size} \
--train.micro_batch_size=${micro_batch_size}"

echo ${cmd}
nsys profile  --trace=cuda,cublas,cudnn,osrt,nvtx \
              --output ${NSYS_OUTPUT} \
              --cpuctxsw=none \
	      --cudabacktrace=all\
              --cuda-memory-usage=true\
               ${cmd}

#nsys profile  --trace=cuda,cublas,cudnn,osrt,nvtx \
#   --gpu-metrics-device=all \
#    --output ${NSYS_OUTPUT} \
#    --sample=cpu \
#    --capture-range=cudaProfilerApi \
#    --cpuctxsw=none \
#    --cuda-memory-usage=true \
#    ${cmd}
