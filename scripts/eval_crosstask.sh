CUDA_VISIBLE_DEVICES=3 python main.py \
    --dropout 0.2 \
    --batch_size 256 \
    --max_traj_len 3 \
    --M 2 \
    --aug_range 0 \
    --attn_heads 32 \
    --num_layers 2 \
    --dataset 'crosstask' \
    --num_action 133 \
    --num_tasks 18 \
    --img_input_dim 512 \
    --text_input_dim 768 \
    --embed_dim 128 \
    --root_dir 'dataset/crosstask/crosstask_release' \
    --train_json 'dataset/crosstask/cross_task_data_False.json' \
    --valid_json 'dataset/crosstask/cross_task_data_True.json' \
    --features_dir 'data/crosstask_features/processed_data' \
    --split 'base' \
    --model_name 'crosstask' \
    --saved_path 'checkpoints' \
    --eval

    