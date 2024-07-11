## Soft Prompt Tuning for Glue Tasks
soft prompt tuning is a method of fine-tuning language models by optimizing input prompts rather than the model itself. It provides a more efficient and flexible way to adapt pre-trained models to specific tasks.

#### Environment Setting

```python
conda create -n [env_name] python=3.11
conda activate [env_name]
pip install -r requirements.txt
pip install -e .
```

#### Train Soft Prompt

```python
mkdir logs
mkdir output
bash scripts/[model_name]/run_prompt_[dataset_name].sh
# e.g. bash scripts/bert_base/run_prompt_sst2.sh
```

#### Reference

- https://github.com/princeton-nlp/LM-BFF
- https://github.com/DengBoCong/prompt-tuning
- https://github.com/ZhengxiangShi/DePT
- https://github.com/WHU-ZQH/PANDA
- https://github.com/AkariAsai/ATTEMPT
- https://github.com/salesforce/Overture/tree/main
- https://github.com/huggingface/transformers