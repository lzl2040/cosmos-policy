NNODES=1
NPROC_PER_NODE=2
BATCH_SIZE=4
DATA_MIX="libero"
STAGE="finetune"
MAX_ACTION_DIM=32
MAX_STATE_dIM=32
DATASET_LEN=5000_0000
ACC_STEP=2
PARENT_DIR="/mnt/wangxiaofa/robot_dataset/lerobot-format"
DCP_PT_PATH=""
USE_STATE=False
CONDITION_FRAMES=4

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --nproc_per_node)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --job_name)
            JOB_NAME="$2"
            shift 2
            ;;
        --use_state)
            USE_STATE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --data_mix)
            DATA_MIX="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --max_action_dim)
            MAX_ACTION_DIM="$2"
            shift 2
            ;;
        --max_state_dim)
            MAX_STATE_DIM="$2"
            shift 2
            ;;
        --cond_frame)
            CONDITION_FRAMES="$2"
            shift 2
            ;;
        --dataset_len)
            DATASET_LEN="$2"
            shift 2
            ;;
        --parent_dir)
            PARENT_DIR="$2"
            shift 2
            ;;
        --dcp_pt_path)
            DCP_PT_PATH="$2"
            shift 2
            ;;
        --acc_step)
            ACC_STEP="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done
export PATH="/opt/conda/envs/lerobot/bin:$PATH"
echo $PATH
ffmpeg
uv run --no-sync --extra cu128 --group libero --python 3.10 \
    python test_ffmpeg.py
which ffmpeg

uv run --no-sync --extra cu128 --group libero --python 3.10 \
  torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py --job_name=$JOB_NAME -- \
  experiment="cosmos_predict2_2b_480p_libero" \
  trainer.grad_accum_iter=$ACC_STEP \
  model.config.dcp_pretrain_path=$DCP_PT_PATH \
  model.config.min_num_conditional_frames=$CONDITION_FRAMES \
  model.config.max_num_conditional_frames=$CONDITION_FRAMES \
  dataloader_train.batch_size=$BATCH_SIZE \
  dataloader_train.dataset.data_mix=$DATA_MIX \
  dataloader_train.dataset.stage=$STAGE \
  dataloader_train.dataset.use_proprio=$USE_STATE \
  dataloader_train.dataset.max_action_dim=$MAX_ACTION_DIM \
  dataloader_train.dataset.max_state_dim=$MAX_STATE_DIM \
  dataloader_train.dataset.dataset_len_one_epoch=$DATASET_LEN \
  dataloader_train.dataset.parent_dir=$PARENT_DIR