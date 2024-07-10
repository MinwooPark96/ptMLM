from transformers import TrainerCallback
import os
import logging
import torch

logger = logging.getLogger(__name__)

class SavePromptCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        self.trainer.save_prompt(output_dir = args.output_dir,
                                checkpoint_name = f"epoch_{int(state.epoch)}")
        