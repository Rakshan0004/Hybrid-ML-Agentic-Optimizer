import os
import torch
import sys
sys.path.insert(0, os.getcwd())
from training.model import ResumeScorerModel
from transformers import LongformerTokenizer

def test_model():
    model_path = "models/final_model/model.pt"
    if not os.path.exists(model_path):
        print("Model not found")
        return
    
    tokenizer = LongformerTokenizer.from_pretrained("models/final_model")
    model = ResumeScorerModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    resume = "Python Developer with experience in Django and React."
    jd1 = "Looking for a Python Developer who knows Django."
    jd2 = "Looking for a Chef who knows how to cook Italian food."
    
    for jd in [jd1, jd2]:
        text = f"{resume} {tokenizer.sep_token} {jd}"
        encoding = tokenizer(text, max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        encoding['global_attention_mask'] = torch.zeros_like(encoding['attention_mask'])
        encoding['global_attention_mask'][0][0] = 1
        
        with torch.no_grad():
            score = model(**encoding)
            print(f"JD: {jd[:30]}... Score: {score.item():.4f}")

if __name__ == "__main__":
    test_model()
