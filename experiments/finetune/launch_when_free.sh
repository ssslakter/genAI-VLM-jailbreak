#!/bin/bash
# Wait for a GPU with enough free memory, then run train.py on it with the given arguments.
# airi.gpu1 is shared: this never takes a card someone is using.
#   ./launch_when_free.sh <log> -- <train.py args...>
set -u
LOG=$1; shift; [ "$1" = "--" ] && shift
NEED_MB=${NEED_MB:-120000}
cd "$(dirname "$0")"
while true; do
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F', ' -v need="$NEED_MB" '$2 >= need {print $1; exit}')
  [ -n "$GPU" ] && break
  sleep 120
done
echo "$(date) starting on GPU $GPU" >> "$LOG"
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=${HF_HOME:-/data/users/epifantsev/hf} HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
exec ../../.pixi/envs/train/bin/python train.py "$@" >> "$LOG" 2>&1
