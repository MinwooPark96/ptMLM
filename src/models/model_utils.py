from .modeling_bert import BertPrompt
from .modeling_roberta import RobertaPrompt

MODELS = {'bert-base-uncased', 'roberta-base', 'bert-large-uncased', 'roberta-large'}
    
def get_model(model_name: str, *args, **kwargs):
        
    if 'bert' in model_name:
        return BertPrompt(*args, **kwargs)
    elif 'roberta' in model_name:
        return RobertaPrompt(*args, **kwargs)
    else:
        raise NotImplementedError(f"[minwoo] model_name = {model_name} is not supported.")