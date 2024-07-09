## Soft Prompt Tuning for Glue Tasks

#### Environment Setting

```python
conda create -n [env_name] python>=3.11
conda activate [env_name]
pip install -r requirements.txt
pip install -e .
```

#### Train Soft Prompt

```python
bash scripts/[model_name]/run_prompt_[dataset_name].sh
# e.g. bash scripts/bert_base/run_prompt_sst2.sh
```

#### Reference
