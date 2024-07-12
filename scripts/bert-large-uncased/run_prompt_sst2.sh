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
    --per_device_train_batch_size 64  \
    --per_device_eval_batch_size 64 \
    --learning_rate 0.0001 \
    --warmup_steps 100 \
    --weight_decay 0.01 \
    --num_train_epochs 20 \
    --output_dir ./output/$model_name_or_path/$dataset_name \
    --pad_to_max_length \
    --pre_seq_len 100 \
    --save_strategy="no" \
    --evaluation_strategy "epoch"\
    --logging_strategy "epoch"\
    --eval_steps 1 \
    --logging_steps 1 \
    --report_to="wandb" \
    --run_name=$model_name_or_path"_"$dataset_name \
    --overwrite_output_dir 1 \
    


