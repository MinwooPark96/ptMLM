import logging
import os
import random
import sys
import transformers
import datasets
import numpy as np
import wandb

from transformers import HfArgumentParser,TrainingArguments
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer
from transformers import set_seed
from transformers import EarlyStoppingCallback, PrinterCallback, TrainerCallback
from transformers.integrations import WandbCallback
 
from src.glue_dataset import GlueDataset
from src.utils import *
from src.models.modeling_bert import BertPrompt
from src.training import Trainer
from src.callbacks import SavePromptCallback
from src.models.model_utils import get_model

from arguments import DataTrainingArguments, ModelArguments

"""
[minwoo] source from :" https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py#L71
"""   

logger = logging.getLogger(__name__)


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
else: #[minwoo] bash
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# [minwoo] logger setting  
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    # handlers=[logging.StreamHandler(sys.stdout)],
    filemode="w",
    filename=os.path.join('logs', f"log_{model_args.model_name_or_path}_{data_args.dataset_name}.txt")
)

if training_args.should_log:
    # The default of training_args.log_level is passive, so we set log level at info here to have that default.
    transformers.utils.logging.set_verbosity_info()

# if training_args.report_to == 'wandb':
#     wandb.init(project='promptTuning_minwoo')

log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
)

# [minwoo] argument 출력하는 부분.
logger.info(f"Training arguments {training_args}")
logger.info(f"Model arguments {model_args}")
logger.info(f"Dataset arguments {data_args}")

# Set seed before initializing model.
set_seed(training_args.seed)

# [minwoo]https://github.com/WHU-ZQH/PANDA/blob/main/p-tuning-v2/tasks/glue/get_trainer.py   
tokenizer = AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
)    
# [minwoo] TODO 데이터가 많아지면, get_data 함수 만들어서 처리.
gluedata = GlueDataset(tokenizer = tokenizer,
        model_args = model_args,
        data_args = data_args, 
        training_args = training_args)

# [minwoo] config source from : https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py
if not gluedata.is_regression:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels = gluedata.num_labels,
        label2id = gluedata.label2id,
        id2label = gluedata.id2label,
        finetuning_task = data_args.dataset_name,
        revision = model_args.model_revision,
    )

else:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=gluedata.num_labels,
        finetuning_task=data_args.dataset_name,
        revision=model_args.model_revision,
    )

model = get_model(
    model_name = model_args.model_name_or_path,
    config = config,
    tokenizer = tokenizer,
    training_args = training_args,
    data_args = data_args,
    model_args = model_args
)

# Initialize our Trainer
# [minwoo] TODO Set wandb
# [minwoo] TODO evaluate.load("glue", data_args.dataset_name) 가 각 dataset에 대하여 어떻게 작동하는지 확인!
# [minwoo] TODO Callback 함수로 주기적으로 저장 및 evaluate 하는 부분 구현.
# [minwoo] TODO compute_loss 를 따로 빼서 trainer 에 넣는것과의 차이 확인?
# Huggingface Trainer : https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = gluedata.train_dataset if training_args.do_train else None, # [minwoo] prepare_input(s) method 에 의하여 잘 처리됨.
    eval_dataset = gluedata.eval_dataset if training_args.do_eval else None,
    compute_metrics = gluedata.compute_metrics, 
    tokenizer = tokenizer,
    data_collator = gluedata.data_collator,
)

save_prompt_callback = SavePromptCallback(trainer)
trainer.add_callback(save_prompt_callback)

def main():
    # [minwoo] Checkpoint detecting 하는 부분.
    last_checkpoint = get_checkpoint(output_dir = training_args.output_dir,
                                                    resume_from_checkpoint = training_args.resume_from_checkpoint)
    
    if last_checkpoint:
        logger.info(
            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
        )
        trainer.load_prompt(prompt_path = last_checkpoint)
    
    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(gluedata.train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(gluedata.train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

      # Evaluation
    if training_args.do_eval:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        dataset_names = [data_args.dataset_name]
        eval_datasets = [gluedata.eval_dataset]
        
        if data_args.task_name == "mnli":
            dataset_names.append("mnli-mm")
            valid_mm_dataset = gluedata.raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, dataset_name in zip(eval_datasets, dataset_names):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if dataset_name == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if dataset_name is not None \
                and "mnli" in dataset_name:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if dataset_name is not None and "mnli" in dataset_name else metrics)

    if training_args.do_predict:
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        dataset_names = [data_args.dataset_name]
        predict_datasets = [gluedata.predict_dataset]
        if data_args.task_name == "mnli":
            dataset_names.append("mnli-mm")
            predict_datasets.append(gluedata.raw_datasets["test_mismatched"])

        for predict_dataset, dataset_name in zip(predict_datasets, dataset_names):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            
            predict_dataset = predict_dataset.remove_columns("label")
            
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if gluedata.is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{dataset_name}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if gluedata.is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = gluedata.label_list[item]
                            writer.write(f"{index}\t{item}\n")


if __name__ == "__main__":
    main()
