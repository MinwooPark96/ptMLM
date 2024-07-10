dataset_name="sst2"
model_name_or_path="bert-large-uncased"

python run_prompt.py   \
    --model_name_or_path $model_name_or_path \
    --task_name glue \
    --dataset_name $dataset_name \
    --do_train \
    --do_eval \
    --do_predict \
    --max_seq_length 128 \
    --per_device_train_batch_size 16  \
    --per_device_eval_batch_size 16 \
    --learning_rate 0.001 \
    --weight_decay 0.0 \
    --num_train_epochs 128 \
    --output_dir ./output/$dataset_name \
    --pad_to_max_length \
    --pre_seq_len 100 \
    --prompt \
    
