NNODES=1
NPROC_PER_NODE=2

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
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

export BASE_DATASETS_DIR="/home/cosmos/.cache/datasets"
export IMAGINAIRE_OUTPUT_ROOT="/home/cosmos/.cache/imaginaire_output"
uv run --no-sync --extra cu128 --group libero --python 3.10 \
  torchrun --nnodes=$NNODES --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py --job_name=$JOB_NAME -- \
  experiment="cosmos_predict2_2b_480p_libero" \
  trainer.grad_accum_iter=8