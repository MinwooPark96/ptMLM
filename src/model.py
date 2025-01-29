from typing import Optional, Union, Tuple

from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from transformers import PretrainedConfig
from transformers import HfArgumentParser,TrainingArguments
from arguments import DataTrainingArguments, ModelArguments

from utils import freeze_params

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

"""
[minwoo] source from :
    1. https://github.com/DengBoCong/prompt-tuning/blob/master/core/prompt_bert.py
    2. https://github.com/thunlp/Prompt-Transferability/blob/main/Prompt-Transferability-1.0/model/PromptBert.py
    3. https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
    4. https://github.com/salesforce/Overture/blob/main/models/modeling_bert.py
    5. https://github.com/QC-LY/Prompt-Tuning-For-Sentiment-Classification/blob/main/prompt_bert.py
"""
        
class MLMPromptModel(nn.Module):

    def __init__(self, 
                config: PretrainedConfig,
                pre_seq_len: int,
                init_type: str = 'random',
                tokenizer = None,
                ):
        
        super(MLMPromptModel,self).__init__()
        
        self.config = config
        
        self.model_name = config._name_or_path
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name) # Pretrained MLM model
        self.normal_embedding_layer = self.model.get_input_embeddings() # word embedding layer of the pretrained model
        
        self.hidden_size = config.hidden_size # e.g. 512, 768, 1024...
        self.finetuning_task = config.finetuning_task # It is from data_args.dataset_name. e.g. sst2 
        
        if tokenizer is None: # Do not recommend to use this way!
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else :
            self.tokenizer = tokenizer
            
        self.mask_ids = torch.tensor([self.tokenizer.mask_token_id]) # token for classification. e.g. [MASK]
    
        self.pre_seq_len = pre_seq_len 

        bert_uncased_map = {'great':2307,'bad':2919,'positive' : 3893,'negative' : 4997,'yes' : 2748,'neutral' : 8699,'no' : 2053,'true' : 2995,'false' : 6270}
        bert_cased_map = {'great':1632,'bad':2213,'positive' : 3112,'negative' : 4366, 'yes' : 4208,'neutral' : 8795,'no' : 1185,'true' : 2276,'false' : 6014}
        roberta_map = {'great':12338, 'bad':1099,'positive' : 22173,'negative' : 2430,'yes' : 4420,'neutral' : 7974,'no' : 117,'true' : 1528,'false' : 3950}
        
        if self.model_name in ['bert-base-uncased','bert-large-uncased']:
            self.map = bert_uncased_map
        elif self.model_name in ['bert-base-cased','bert-large-cased']:
            self.map = bert_cased_map
        elif self.model_name in ['roberta-base','roberta-large']:
            self.map = roberta_map
        else:
            raise NotImplementedError(f"[minwoo] word_map for {self.model_name} is not supported yet.")
        
        # [minwoo] https://github.com/salesforce/Overture/blob/main/soft_prompts.py -> 초기화 방식 참고.
        if init_type == 'random':
            # self.soft_prompts = nn.Parameter(self.soft_prompts, requires_grad=True)  #https://github.com/salesforce/Overture/blob/main/soft_prompts.py
            self.soft_prompt = nn.Parameter(torch.randn(self.pre_seq_len, self.hidden_size)) # [minwoo] size = (pre_seq_len, hidden_size)
        elif init_type == 'zero':
            self.soft_prompt = nn.Parameter(torch.zeros(self.pre_seq_len, self.hidden_size))
        else:
            raise NotImplementedError(f"[minwoo] init_type = {init_type} is not supported.")

        if self.finetuning_task == 'stsb':
            NotImplementedError("[minwoo] STSB task is not supported yet... It is regression task.")
        
        # [minwoo] freeze model
        freeze_params(self.model)
    
    def get_soft_prompt(self):
        """Return the soft prompt."""
        return self.soft_prompt

    def get_word_embedding_layer(self):
        """Return the word embedding layer."""
        return self.normal_embedding_layer

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
             ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:

        """
        [minwoo]
            I assume that the arguments does not prepend additional tokens for target [mask] and [soft_prompt].
        """

        batch_size = input_ids.shape[0] #[minwoo] batch_size 가 edge 부분에서 달라질 수 있으므로, 여기서 계산하는 것이 맞음.
        # mask_ids = torch.stack([self.mask_ids for _ in range(batch_size)]).to(input_ids.device)
        mask_ids = self.mask_ids.expand(batch_size, -1).to(input_ids.device)
        
        # [minwoo] each_embeddings.shape = (batch_size, each , hidden_size)
        mask_embeddings = self.normal_embedding_layer(mask_ids)
        soft_embeddings = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1) 
        text_embeddings = self.normal_embedding_layer(input_ids)
        
        mask_attention_mask = torch.ones(batch_size, 1).to(input_ids.device)
        soft_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(input_ids.device)
        
        input_embeddings = torch.cat([mask_embeddings, soft_embeddings,text_embeddings], dim = 1)
        attention_mask = torch.cat([mask_attention_mask, soft_attention_mask, attention_mask],dim=1)
        
        assert attention_mask.shape[1] == input_embeddings.shape[1] # == 1 + pre_seq_len + text_seq_len
        
        model_outputs = self.model( # AutoModelForMaskedLM
            attention_mask = attention_mask,
            inputs_embeds = input_embeddings,
        ) # -> MaskedLMOutput(loss = None, logits = tensor(batch, total_length, vacab_size))
        
        # [minwoo] logits.shape = (batch, total_length, vocab_size)
        logits = model_outputs.logits
        mask_logits = logits[:,0]

        # [minwoo] score.shape = (batch, Class_num)         
        #   TODO -> https://github.com/DengBoCong/prompt-tuning/blob/master/core/prompt_bert.py#L107 를 참고하여 if else 제거
        if self.finetuning_task in ['sst2', 'imdb']:
            score = torch.cat([mask_logits[:,self.map['positive']].unsqueeze(1), mask_logits[:,self.map['negative']].unsqueeze(1)],dim = 1)
            # score = torch.cat([mask_logits[:,self.map['great']].unsqueeze(1), mask_logits[:,self.map['bad']].unsqueeze(1)],dim = 1)
        elif self.finetuning_task in ['mnli']:
            score = torch.cat([mask_logits[:,self.map['yes']].unsqueeze(1), mask_logits[:,self.map['neutral']].unsqueeze(1), mask_logits[:,self.map['no']].unsqueeze(1)],dim = 1)
        elif self.finetuning_task in ['qqp', 'mrpc']:
            score = torch.cat([mask_logits[:,self.map['true']].unsqueeze(1), mask_logits[:,self.map['false']].unsqueeze(1)], dim = 1)
        else :
            NotImplementedError(f"[minwoo] {self.finetuning_task} is not supported yet...")
        
        loss = None
        if labels is not None: # [minwoo] It would be called in training phase.
            # [minwoo] ignore_index = -100 으로 default setting 되어 있음
            #  see : https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(score, labels.view(-1)) 
        
        outputs = (score,)
        
        return ((loss,)+ outputs) if loss is not None else outputs

            
if __name__ == '__main__':
    
    from utils import print_params_only_requires_grad_true
    from src.datasets.glue import GlueDataset
    from transformers import Trainer

    # python modeling_bert.py --task_name glue --dataset_name sst2 --model_name_or_path bert-base-uncased --output_dir ./output
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    config = AutoConfig.from_pretrained('bert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    model = MLMPromptModel(
        config = config,
        pre_seq_len = 5)
    
    print_params_only_requires_grad_true(model)
    
    print(tokenizer.encode(('positive negative yes neutral no true false')))
