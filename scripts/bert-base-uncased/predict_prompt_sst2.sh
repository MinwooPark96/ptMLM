dataset_name="sst2"
model_name_or_path="bert-base-uncased"

python run_prompt.py   \
    --model_name_or_path $model_name_or_path \
    --task_name glue \
    --dataset_name $dataset_name \
    --do_predict \
    --max_seq_length 128 \
    --per_device_eval_batch_size 64 \
    --output_dir ./output/$model_name_or_path/$dataset_name \
    --pad_to_max_length \
    --pre_seq_len 100 \
    --prompt \
    # --overwrite_output_dir 1 \
    


