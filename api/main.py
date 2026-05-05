"""
FastAPI Backend for JobFit Agent.

Endpoints:
  POST /score          — Score a resume against a JD (stateless)
  WS   /ws/agent       — WebSocket for the agentic chat interface
  GET  /health         — Health check
"""
import os
import sys
import json
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()  # Load .env file

# Add project root to path so we can import training module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import extract_text_from_latex

app = FastAPI(title="JobFit Agent API")

# Allow CORS for React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Model Loading (Lazy)
# ──────────────────────────────────────────────
ml_model = None
ml_tokenizer = None

def load_ml_model():
    global ml_model, ml_tokenizer
    if ml_model is not None:
        return
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "final_model", "model.pt")
    if not os.path.exists(model_path):
        print("WARNING: Trained model not found at", model_path)
        return
    
    from training.model import ResumeScorerModel
    from transformers import LongformerTokenizer
    
    print("Loading ML model...")
    tokenizer_path = os.path.join(os.path.dirname(model_path))
    ml_tokenizer = LongformerTokenizer.from_pretrained(tokenizer_path)
    ml_model = ResumeScorerModel()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    ml_model.load_state_dict(state_dict)
    ml_model.eval()
    print("ML model loaded successfully!")


# ──────────────────────────────────────────────
# REST Endpoint: Score Resume
# ──────────────────────────────────────────────
class ScoreRequest(BaseModel):
    resume_text: str
    job_description: str

@app.post("/score")
async def score_resume_endpoint(request: ScoreRequest):
    load_ml_model()
    if ml_model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run training/train.py first.")
    
    # Strip LaTeX commands before scoring (model was trained on plain text)
    clean_resume = extract_text_from_latex(request.resume_text)
    
    text = f"{clean_resume} {ml_tokenizer.sep_token} {request.job_description}"
    encoding = ml_tokenizer(
        text,
        max_length=4096,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    encoding['global_attention_mask'] = torch.zeros_like(encoding['attention_mask'])
    encoding['global_attention_mask'][0][0] = 1
    
    with torch.no_grad():
        score = ml_model(**encoding)
    
    return {"confidence_score": round(float(score.item()) * 100, 2)}


# ──────────────────────────────────────────────
# WebSocket: Agentic Chat Interface
# ──────────────────────────────────────────────
@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    await websocket.accept()
    
    from agent.orchestrator import AgentOrchestrator
    
    # Prioritize Gemini key as it's the most reliable for the user
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        await websocket.send_json({"type": "error", "content": "No API Key found in .env file (GEMINI_API_KEY or OPENROUTER_API_KEY)"})
        await websocket.close()
        return
    
    load_ml_model()
    
    orchestrator = AgentOrchestrator(
        api_key=api_key,
        ml_model=ml_model,
        ml_tokenizer=ml_tokenizer
    )
    
    session_initialized = False
    
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")
            
            if msg_type == "init":
                # Initialize session with LaTeX code and JD
                latex_code = data.get("latex_code", "")
                job_description = data.get("job_description", "")
                
                if not latex_code or not job_description:
                    await websocket.send_json({"type": "error", "content": "Both latex_code and job_description are required."})
                    continue
                
                orchestrator.start_session(latex_code, job_description)
                session_initialized = True
                await websocket.send_json({"type": "system", "content": "Session initialized. You can now chat with the agent to improve your resume."})
                
                # Send the initial LaTeX content
                await websocket.send_json({"type": "latex_update", "content": latex_code})
            
            elif msg_type == "message":
                if not session_initialized:
                    await websocket.send_json({"type": "error", "content": "Session not initialized. Send an 'init' message first."})
                    continue
                
                user_message = data.get("content", "")
                if not user_message:
                    continue
                
                # Stream agent events back to the frontend
                print(f"DEBUG: Received message from user: {user_message}")
                async for response in orchestrator.process_message(user_message):
                    print(f"DEBUG: Sending response to frontend: {response['type']}")
                    await websocket.send_json(response)
            
            else:
                await websocket.send_json({"type": "error", "content": f"Unknown message type: {msg_type}"})
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": ml_model is not None}
