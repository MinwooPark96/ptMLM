from typing import Optional, Union, Tuple

from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from transformers import PretrainedConfig
from transformers import HfArgumentParser,TrainingArguments
from arguments import DataTrainingArguments, ModelArguments

from utils import freeze_params

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

        
class RobertaPrompt(RobertaPreTrainedModel):

    def __init__(self, 
                config: PretrainedConfig,
                tokenizer = None,
                training_args: TrainingArguments = None,
                data_args: DataTrainingArguments = None,
                model_args: ModelArguments = None):
        
        super().__init__(config)
        self.config = config
        self.training_args = training_args
        self.data_args = data_args
        self.model_args = model_args
        
        self.model = AutoModelForMaskedLM.from_pretrained(config._name_or_path) # Pretrained MLM model
        self.normal_embedding_layer = self.model.get_input_embeddings() # word embedding layer of the pretrained model
        self.hidden_size = config.hidden_size # e.g. 512, 768, 1024...
        self.finetuning_task = config.finetuning_task # It is from data_args.dataset_name. e.g. sst2 
        
        if tokenizer is None: # Do not recommend to use this way!
            self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
        else :
            self.tokenizer = tokenizer
            
        self.mask_ids = torch.tensor([self.tokenizer.mask_token_id]) # token for classification. e.g. [MASK]
    
        self.pre_seq_len = model_args.pre_seq_len 
        
        # [minwoo] https://github.com/salesforce/Overture/blob/main/soft_prompts.py -> 초기화 방식 참고.
        if model_args.init_type == 'random':
            self.soft_prompt = nn.Parameter(torch.randn(self.pre_seq_len, self.hidden_size)) # [minwoo] size = (pre_seq_len, hidden_size)
        elif model_args.init_type == 'zero':
            self.soft_prompt = nn.Parameter(torch.zeros(self.pre_seq_len, self.hidden_size))
        else:
            raise NotImplementedError(f"[minwoo] model_args.init_type = {model_args.init_type} is not supported.")

         
        if self.finetuning_task == 'stsb':
            NotImplementedError("[minwoo] STSB task is not supported yet... It is regression task.")
        
        # [minwoo] freeze model
        freeze_params(self.model)
        
        # Initialize weights and apply final processing
        self.post_init()

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

        """                
        SA (Sentiment Analysis)
            IMDB: positive, negative 
            SST-2: positive, negative
                Roberta : 22173, 2430

        NLI (Natural Language Inference)**: 
            MNLI: yes, neutral, no
                Roberta : 4420, 7974, 117
    
    
        PI (Paraphrase Identification)**:
            QQP: true, false
            MRPC: true, false
                Roberta : 1528, 3950
        
        """

        # [minwoo] score.shape = (batch, Class_num)         
        #   TODO -> https://github.com/DengBoCong/prompt-tuning/blob/master/core/prompt_bert.py#L107 를 참고하여 if else 제거
        if self.finetuning_task in ['sst2', 'imdb']:
            score = torch.cat([mask_logits[:,3893].unsqueeze(1), mask_logits[:,4997].unsqueeze(1)],dim = 1)
        elif self.finetuning_task in ['mnli']:
            score = torch.cat([mask_logits[:,2748].unsqueeze(1), mask_logits[:,8699].unsqueeze(1), mask_logits[:,2053].unsqueeze(1)],dim = 1)
        elif self.finetuning_task in ['qqp', 'mrpc']:
            score = torch.cat([mask_logits[:,2995].unsqueeze(1), mask_logits[:,6270].unsqueeze(1)], dim = 1)
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

    # python modeling_roberta.py --task_name glue --dataset_name qqp --model_name_or_path roberta-base --output_dir ./output
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        
    model = RobertaPrompt(
        config = config,
        tokenizer = None,
        training_args = training_args,
        data_args = data_args,
        model_args = model_args
        )
    
    print_params_only_requires_grad_true(model)
    
    print(tokenizer.encode(('positive negative yes neutral no true false')))
    # [0, 22173, 2430, 4420, 7974, 117, 1528, 3950, 2]