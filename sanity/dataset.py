import sys
sys.path.append('..')

from src.datasets.glue import GlueDataset
from src.datasets.glue import task_to_keys as glue_task_to_keys
from src.datasets.superglue import SuperGlueDataset
from src.datasets.superglue import task_to_keys as superglue_task_to_keys

from transformers import AutoTokenizer

DATASET = ['glue', 'superglue']
TASKS = list(glue_task_to_keys.keys()) + list(superglue_task_to_keys.keys())

if __name__ == '__main__':
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')    
    
    # for glue_data in glue_task_to_keys:
    #     dataset = GlueDataset(
    #         dataset_name = glue_data,
    #         tokenizer = tokenizer,
    #         pad_to_max_length = True,
    #         max_seq_length = 128)
    
    for superglue_data in superglue_task_to_keys:
        dataset = SuperGlueDataset(
            dataset_name = superglue_data,
            model_name='bert',
            tokenizer = tokenizer,
            pad_to_max_length = True,
            max_seq_length = 128)
    
    print(dataset)


