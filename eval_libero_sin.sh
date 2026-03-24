CKPT_PATH=""
LOG_ID=""
TASK_SUITE=""
NUM_PER_TASK=20
MAX_ACTION_DIM=7
MAX_STATE_DIM=8
ACTION_STEP=16

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ckpt_path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --log_id)
            LOG_ID="$2"
            shift 2
            ;;
        
        --task_suite)
            TASK_SUITE="$2"
            shift 2
            ;;
        --num_per_task)
            NUM_PER_TASK="$2"
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
        --action_step)
            ACTION_STEP="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done
export PATH=/home/aiscuser/.conda/envs/lerobot/bin:$PATH
export LD_LIBRARY_PATH=/home/aiscuser/.conda/envs/lerobot/lib:$LD_LIBRARY_PATH
export LIBERO_CONFIG_PATH=/mnt/wangxiaofa/cosmos_policy_exp/cosmos_policy
# export COSMOS_SMOKE=1
uv run --no-sync --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path $CKPT_PATH \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --use_third_person_image True \
    --num_third_person_images 1 \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path /mnt/wangxiaofa/cosmos_policy_exp/cosmos_policy/libero_stats.json \
    --t5_text_embeddings_path /mnt/wangxiaofa/cosmos_policy_exp/cosmos_policy/t5_embeddings_libero.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps $ACTION_STEP \
    --task_suite_name $TASK_SUITE \
    --num_trials_per_task $NUM_PER_TASK \
    --device 0 \
    --local_log_dir /mnt/wangxiaofa/cosmos_policy_exp/cosmos_policy/libero/logs/ \
    --randomize_seed False \
    --data_collection False \
    --available_gpus "0,1,2,3" \
    --use_parallel_inference False \
    --seed 195 \
    --use_variance_scale False \
    --deterministic True \
    --run_id_note $LOG_ID \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression False \
    --flip_images True \
    --num_denoising_steps_action 5 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --max_action_dim $MAX_ACTION_DIM \
    --max_state_dim $MAX_STATE_DIM
