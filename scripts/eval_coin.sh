CUDA_VISIBLE_DEVICES=1 python main.py \
    --dropout 0.2 \
    --batch_size 256 \
    --max_traj_len 4 \
    --M 2 \
    --aug_range 0 \
    --attn_heads 32 \
    --num_layers 2 \
    --dataset 'coin' \
    --num_action 778 \
    --num_tasks 180 \
    --img_input_dim 512 \
    --text_input_dim 768 \
    --embed_dim 128 \
    --root_dir 'dataset/coin' \
    --train_json 'dataset/coin/coin_train.json' \
    --valid_json 'dataset/coin/coin_valid.json' \
    --features_dir '/home/wenliang/data/coin_HowTo100_feature' \
    --model_name 'coin' \
    --saved_path 'checkpoints' \
    --eval