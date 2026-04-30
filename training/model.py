import torch
import torch.nn as nn
from transformers import LongformerModel, LongformerConfig, PreTrainedModel

class ResumeScorerModel(PreTrainedModel):
    config_class = LongformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output 0-1 for normalized score
        )
        
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, labels=None):
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask
        )
        
        # Use the [CLS] token representation (first token)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.regressor(pooled_output).squeeze(-1)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            
        return ((loss, logits) if loss is not None else logits)

import os
os.environ["HF_HOME"] = "D:/Coding/resume-reviewer/.hf_cache"

def get_model_and_tokenizer():
    from transformers import LongformerTokenizer
    model_name = "allenai/longformer-base-4096"
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = ResumeScorerModel.from_pretrained(model_name, use_safetensors=True)
    return model, tokenizer
