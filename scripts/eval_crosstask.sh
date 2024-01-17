CUDA_VISIBLE_DEVICES=2 python main.py \
    --dropout 0.2 \
    --batch_size 256 \
    --max_traj_len 4 \
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
    --description_type "cot" \
    --features_dir '/home/wenliang/data/processed_data_crosstask' \
    --split 'base' \
    --description_type "cot" \
    --model_name 'crosstask' \
    --saved_path 'checkpoints' \
    --eval

    