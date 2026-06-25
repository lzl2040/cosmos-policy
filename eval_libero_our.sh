uv run --no-sync --extra cu128 --group libero --python 3.10 \
  python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path "/home/cosmos/.cache/cosmos_policy/libero/0618_mean_std_cosmos_ace_action_encoder/iter_000010000/model" \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True \
    --use_proprio True \
    --use_third_person_image True \
    --num_third_person_images 2 \
    --normalize_proprio True \
    --unnormalize_actions True \
    --dataset_stats_path /home/cosmos/.cache/lerobot_data/libero_stats.json \
    --t5_text_embeddings_path /home/cosmos/.cache/lerobot_data/t5_embeddings_libero.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 \
    --num_open_loop_steps 16 \
    --task_suite_name libero_spatial \
    --num_trials_per_task 20 \
    --device 0 \
    --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
    --randomize_seed False \
    --data_collection False \
    --available_gpus "0,1,2,3" \
    --use_parallel_inference False \
    --seed 195 \
    --use_variance_scale False \
    --deterministic True \
    --run_id_note cosmos-ace-vae-feats-world-chkpt24000--5stepAct--seed195--deterministic-debug \
    --ar_future_prediction False \
    --ar_value_prediction False \
    --use_jpeg_compression False \
    --flip_images True \
    --num_denoising_steps_action 10 \
    --num_denoising_steps_future_state 1 \
    --num_denoising_steps_value 1 \
    --max_action_dim 32 \
    --max_state_dim 32

# uv run --no-sync --extra cu128 --group libero --python 3.10 \
#   python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
#     --config cosmos_predict2_2b_480p_libero__inference_only \
#     --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
#     --config_file cosmos_policy/config/config.py \
#     --use_wrist_image True \
#     --use_proprio False \
#     --normalize_proprio True \
#     --unnormalize_actions True \
#     --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
#     --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
#     --trained_with_image_aug True \
#     --chunk_size 16 \
#     --num_open_loop_steps 16 \
#     --task_suite_name libero_object \
#     --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
#     --randomize_seed False \
#     --data_collection False \
#     --available_gpus "0,1,2,3" \
#     --seed 195 \
#     --use_variance_scale False \
#     --deterministic True \
#     --run_id_note chkpt45000--5stepAct--seed195--deterministic \
#     --ar_future_prediction False \
#     --ar_value_prediction False \
#     --use_jpeg_compression True \
#     --flip_images True \
#     --num_denoising_steps_action 5 \
#     --num_denoising_steps_future_state 1 \
#     --num_denoising_steps_value 1