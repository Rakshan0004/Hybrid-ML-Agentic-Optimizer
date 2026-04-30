import os
os.environ["HF_HOME"] = "D:/Coding/resume-reviewer/.hf_cache"

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from model import get_model_and_tokenizer
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ResumeDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.resume_texts = df['resume_text'].tolist()
        self.jd_texts = df['job_description'].tolist()
        self.labels = df['total_score'].tolist()
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Concatenate resume and JD with [SEP]
        text = f"{self.resume_texts[idx]} {self.tokenizer.sep_token} {self.jd_texts[idx]}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        
        # Longformer needs global attention on the [CLS] token
        item['global_attention_mask'] = torch.zeros_like(item['attention_mask'])
        item['global_attention_mask'][0] = 1
        
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.squeeze()
    mse = mean_squared_error(labels, logits)
    mae = mean_absolute_error(labels, logits)
    rmse = np.sqrt(mse)
    return {"rmse": rmse, "mae": mae}

def train():
    print("Loading data...")
    df = pd.read_parquet("data/train_raw.parquet")
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.9 * len(df))
    train_df = df[:train_size]
    val_df = df
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")
    
    model, tokenizer = get_model_and_tokenizer()
    
    train_dataset = ResumeDataset(train_df, tokenizer)
    val_dataset = ResumeDataset(val_df, tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./models/longformer-resume-scorer",
        num_train_epochs=5,
        per_device_train_batch_size=1, # Small batch size for Longformer on 12GB VRAM
        gradient_accumulation_steps=16, # Effective batch size = 16
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="rmse",
        greater_is_better=False,
        fp16=True, # Use mixed precision for GPU
        logging_steps=10,
        push_to_hub=False,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model("./models/final_model")
    tokenizer.save_pretrained("./models/final_model")

if __name__ == "__main__":
    train()
