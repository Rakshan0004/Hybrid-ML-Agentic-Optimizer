from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from training.model import get_model_and_tokenizer
import os

app = FastAPI(title="Resume Reviewer API")

# Lazy loading of model to avoid issues during development
model = None
tokenizer = None

class ScoreRequest(BaseModel):
    resume_text: str
    job_description: str

@app.post("/score")
async def score_resume(request: ScoreRequest):
    global model, tokenizer
    if model is None:
        # Load model from final_model if it exists
        if os.path.exists("./models/final_model"):
            from training.model import ResumeScorerModel
            from transformers import LongformerTokenizer
            tokenizer = LongformerTokenizer.from_pretrained("./models/final_model")
            model = ResumeScorerModel.from_pretrained("./models/final_model")
            model.eval()
        else:
            raise HTTPException(status_code=503, detail="Model not trained yet")
    
    text = f"{request.resume_text} {tokenizer.sep_token} {request.job_description}"
    encoding = tokenizer(
        text,
        max_length=4096,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Global attention on [CLS]
    encoding['global_attention_mask'] = torch.zeros_like(encoding['attention_mask'])
    encoding['global_attention_mask'][0][0] = 1
    
    with torch.no_grad():
        score = model(**encoding)
    
    return {"confidence_score": float(score.item()) * 100}

@app.get("/health")
async def health():
    return {"status": "ok"}
