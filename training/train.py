import os
os.environ["HF_HOME"] = "./.hf_cache"

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from model import get_model_and_tokenizer
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

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


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            loss, logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                global_attention_mask=global_attention_mask,
                labels=labels
            )
            
            total_loss += loss.item()
            all_preds.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    rmse = math.sqrt(mean_squared_error(all_labels, all_preds))
    mae = mean_absolute_error(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, rmse, mae


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    df = pd.read_parquet("data/train_raw.parquet")
    
    # Shuffle and split 90/10
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.9 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    print(f"Training on {len(train_df)} samples, validating on {len(val_df)} samples.")
    print(f"Score range in data: {df['total_score'].min():.3f} - {df['total_score'].max():.3f}")
    
    model, tokenizer = get_model_and_tokenizer()
    model.to(device)
    
    train_dataset = ResumeDataset(train_df, tokenizer)
    val_dataset = ResumeDataset(val_df, tokenizer)
    
    # Batch size 1 + gradient accumulation = effective batch size of 8
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    num_epochs = 5
    accumulation_steps = 8
    best_rmse = float('inf')
    patience = 2
    patience_counter = 0
    
    # Disable mixed precision for GPU as it crashes Longformer on Windows
    scaler = None
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            global_attention_mask = batch['global_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            if scaler:
                with torch.amp.autocast("cuda"):
                    loss, logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        global_attention_mask=global_attention_mask,
                        labels=labels
                    )
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    labels=labels
                )
                loss = loss / accumulation_steps
                loss.backward()
            
            total_train_loss += loss.item() * accumulation_steps
            
            if (step + 1) % accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            
            if (step + 1) % 10 == 0 or step == 0:
                print(f"  Epoch {epoch+1} | Step {step+1}/{len(train_loader)} | Loss: {loss.item() * accumulation_steps:.4f}")
        
        # Final gradient step if not aligned
        if (step + 1) % accumulation_steps != 0:
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Evaluate
        val_loss, val_rmse, val_mae = evaluate(model, val_loader, device)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val MAE: {val_mae:.4f}")
        
        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            os.makedirs("./models/final_model", exist_ok=True)
            torch.save(model.state_dict(), "./models/final_model/model.pt")
            tokenizer.save_pretrained("./models/final_model")
            print(f"  ✓ New best model saved (RMSE: {best_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("  Early stopping triggered.")
                break
        print()
    
    print(f"\nTraining complete! Best RMSE: {best_rmse:.4f}")
    print("Model saved to ./models/final_model/")


if __name__ == "__main__":
    train()
