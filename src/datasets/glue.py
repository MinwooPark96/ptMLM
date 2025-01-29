from datasets.load import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import numpy as np
import logging
import evaluate
from typing import Optional
"""
[minwoo] source from : https://github.com/WHU-ZQH/PANDA/blob/main/p-tuning-v2/tasks/glue/dataset.py
huggigface run_glue.py : https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py#L71
"""

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)

class GlueDataset:
    def __init__(self, 
                 dataset_name: Optional[str],
                 tokenizer: AutoTokenizer,
                 pad_to_max_length: bool = True,
                 max_seq_length: int = 128,
                 do_train: bool = True,
                 max_train_samples: Optional[int] = None,
                 do_eval: bool = True,
                 max_eval_samples: Optional[int] = None,
                 do_predict: bool = True,
                 max_predict_samples: Optional[int] = None,
                 cache_dir: Optional[str] = None,
                 overwrite_cache: bool = False,
                 *args,
                **kwargs
                 ) -> None:
        """
        
        [minwoo] args:
            1. tokenizer: AutoTokenizer 
            2. model_args:
                - cache_dir: str
            3. data_args: 
                - dataset_name: str
                    e.g. "cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"
                
                - max_seq_length: int
                    e.g. 128
                
                - pad_to_max_length: bool
                    
                - max_train_samples: Optional[int]
                        
                - max_eval_samples: Optional[int]
                
                - max_predict_samples: Optional[int]
                
                - overwrite_cache: bool
                    raw_datasets.map(...,
                        load_from_cache_file = not data_args.overwrite_cache,
                        ...)
            4. training_args:
                - do_train: bool
                - do_eval: bool
                - do_predict: bool
                - fp16: bool 
                    if True : DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
                    else : default_data_collator
                        see https://github.com/huggingface/transformers/blob/main/src/transformers/data/data_collator.py#L74
                        
        """
        
        super().__init__()
        
        raw_datasets = load_dataset("glue", dataset_name, trust_remote_code=True)
        self.tokenizer = tokenizer
        
        #labels
        self.is_regression = dataset_name == "stsb" # [minwoo] stsb 는 regression task
        
        if not self.is_regression:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        
        else:
            self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[dataset_name]

        # Padding strategy
        if pad_to_max_length: #[minwoo] 이쪽을 사용하는 듯?
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(max_seq_length, tokenizer.model_max_length)

        raw_datasets = raw_datasets.map(
            self.preprocess_function,
            batched=True,
            load_from_cache_file = not overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        if do_train:
            self.train_dataset = raw_datasets["train"]
            if max_train_samples is not None:
                self.train_dataset = self.train_dataset.select(range(max_train_samples))

        if do_eval:
            self.eval_dataset = raw_datasets["validation_matched" if dataset_name == "mnli" else "validation"]
            if max_eval_samples is not None:
                self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))

        if do_predict:
            self.predict_dataset = raw_datasets["test_matched" if dataset_name == "mnli" else "test"]
            
            if max_predict_samples is not None:
                self.predict_dataset = self.predict_dataset.select(range(max_predict_samples))

        # [minwoo] self.metric = load_metric("glue", data_args.dataset_name) -> warning 수정
        if dataset_name is not None:
            self.metric = evaluate.load("glue", 
                                        dataset_name, 
                                        cache_dir=cache_dir)
        elif self.is_regression:
            self.metric = evaluate.load("mse", 
                                        cache_dir = cache_dir)
        else:
            self.metric = evaluate.load("accuracy", 
                                        cache_dir = cache_dir)
        
        # [minwoo] else 는 hf Line 525 보고 추가함.
        if pad_to_max_length:
            """
            data_colloar(
                dataset : Dataset(features : list[list[Any]) 
            ) 
                -> Dict[str,torch.Tensor]
            """
            self.data_collator = default_data_collator
        
        # elif training_args.fp16:
            # self.data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
            
            """
	            1.	input_ids: 원래의 텍스트 시퀀스에서 일부 토큰이 [MASK] 토큰으로 대체된 시퀀스입니다.
	            2.  labels: 원래의 텍스트 시퀀스를 나타내며, 모델이 예측해야 할 목표(target)입니다. [MASK] 토큰이 아닌 위치는 -100으로 설정되어 손실 계산에서 무시됩니다.
            """
        
        else : 
            self.data_collator = None

    # [minwoo] init에서 자동으로 호출됨.
    def preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples[self.sentence1_key],) if self.sentence2_key is None else (examples[self.sentence1_key], examples[self.sentence2_key])
        )       
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
        return result
    
    """
    [minwoo] 
    의 Line 510 과 거의 유사 
    """
    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if self.is_regression else np.argmax(preds, axis=1)
        
        if self.data_args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        # [minwoo] glue task 에서 이 아래로 내려올 일 없을 듯?
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}


    