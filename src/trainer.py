import logging
import torch
from transformers import Trainer as HF_Trainer

logger = logging.getLogger(__name__)

class Trainer(HF_Trainer):
    def save_prompt(self, output_dir = None, checkpoint_name = None):
        if not output_dir:
            output_dir = self.args.output_dir
        
        prompt_state_dict = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                prompt_state_dict[name] = param.detach().cpu()
        if not checkpoint_name:
            torch.save(prompt_state_dict, f"{output_dir}/prompt.bin")
        else :
            torch.save(prompt_state_dict, f"{output_dir}/{checkpoint_name}_prompt.bin")
    
    def get_soft_prompt(self):
        return self.model.get_soft_prompt()
    
    def load_prompt(self, prompt_path):
        prompt_state_dict = torch.load(prompt_path)
        for name, param in self.model.named_parameters():
            if name in prompt_state_dict:
                param.data = prompt_state_dict[name].to(param.device)
        


