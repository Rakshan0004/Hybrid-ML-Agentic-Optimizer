"""
Agent Tools - Defined as callable functions with schemas for Gemini Function Calling.

The LLM will decide WHEN and HOW to use these tools during the improvement loop.
"""
import re
import json
from typing import Optional

# ──────────────────────────────────────────────
# Tool Implementations
# ──────────────────────────────────────────────

class LatexDocument:
    """In-memory representation of the user's LaTeX resume. Shared state for all tools."""
    def __init__(self, content: str):
        self.content = content
        self.history: list[str] = [content]  # version history
    
    def get_content(self) -> str:
        return self.content
    
    def update_content(self, new_content: str):
        self.history.append(new_content)
        self.content = new_content
    
    def undo(self) -> str:
        if len(self.history) > 1:
            self.history.pop()
            self.content = self.history[-1]
        return self.content


def read_latex(doc: LatexDocument) -> str:
    """Read the current LaTeX source code."""
    return doc.get_content()


def edit_latex(doc: LatexDocument, new_content: str) -> str:
    """Replace the entire LaTeX document with new content."""
    doc.update_content(new_content)
    return f"LaTeX document updated successfully. New length: {len(new_content)} chars."


def edit_latex_section(doc: LatexDocument, old_text: str, new_text: str) -> str:
    """Replace a specific section of the LaTeX document (search & replace)."""
    current = doc.get_content()
    if old_text not in current:
        return f"ERROR: Could not find the specified text in the document. Make sure you copy the exact text."
    
    updated = current.replace(old_text, new_text, 1)
    doc.update_content(updated)
    return f"Successfully replaced section. Changed {len(old_text)} chars -> {len(new_text)} chars."


def extract_text_from_latex(latex_code: str) -> str:
    """Extract readable text from LaTeX for scoring purposes."""
    text = re.sub(r'%.*?\n', '\n', latex_code)
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[.*?\])?\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\begin\{.*?\}', '', text)
    text = re.sub(r'\\end\{.*?\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def score_resume(resume_text: str, job_description: str, model=None, tokenizer=None) -> dict:
    """
    Score a resume against a job description using the trained Longformer model.
    Returns the confidence score as a percentage.
    """
    if model is None or tokenizer is None:
        return {"score": -1, "error": "Model not loaded. Train the model first."}
    
    import torch
    text = f"{resume_text} {tokenizer.sep_token} {job_description}"
    encoding = tokenizer(
        text,
        max_length=4096,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    encoding['global_attention_mask'] = torch.zeros_like(encoding['attention_mask'])
    encoding['global_attention_mask'][0][0] = 1
    
    device = next(model.parameters()).device
    encoding = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        result = model(**encoding)
    
    score = float(result.item()) * 100
    return {"score": round(score, 2)}


# ──────────────────────────────────────────────
# Tool Schemas for Gemini Function Calling
# ──────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "read_latex",
        "description": """Read the current full LaTeX source code of the resume.

WHEN TO USE:
- At the START of every session before making any changes
- After the user asks about what's in their resume
- When you need to verify the current state of the document before making edits
- Before calling score_resume, to understand what you're scoring

RETURNS: The full LaTeX source code as a string.""",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "edit_latex_section",
        "description": """Replace a specific section of the LaTeX resume using search-and-replace.

WHEN TO USE:
- For targeted changes like rewriting a single bullet point, adding a skill, or modifying a section heading
- When you want to change specific wording or add keywords from the job description
- This is your PRIMARY editing tool — prefer this over edit_latex for most changes

WHEN NOT TO USE:
- Do NOT use this if you want to rewrite the entire document — use edit_latex instead
- Do NOT guess the old_text — always call read_latex first to see the exact content

IMPORTANT:
- old_text MUST match the document EXACTLY (including whitespace and LaTeX commands)
- If the match fails, the tool will return an error. Read the document again and try with the correct text.""",
        "parameters": {
            "type": "object",
            "properties": {
                "old_text": {
                    "type": "string",
                    "description": "The exact text currently in the LaTeX document that you want to replace. Must match character-for-character."
                },
                "new_text": {
                    "type": "string",
                    "description": "The new text to replace the old text with. Must be valid LaTeX."
                }
            },
            "required": ["old_text", "new_text"]
        }
    },
    {
        "name": "edit_latex",
        "description": """Replace the ENTIRE LaTeX document with completely new content.

WHEN TO USE:
- Only for major structural rewrites (e.g., reorganizing the entire document layout)
- When the user explicitly asks for a full rewrite

WHEN NOT TO USE:
- Do NOT use this for small changes — use edit_latex_section instead
- Do NOT use this unless absolutely necessary, as it replaces everything""",
        "parameters": {
            "type": "object",
            "properties": {
                "new_content": {
                    "type": "string",
                    "description": "The complete new LaTeX document content. Must be a full, valid LaTeX document."
                }
            },
            "required": ["new_content"]
        }
    },
    {
        "name": "score_resume",
        "description": """Score the current resume against the job description using the trained ML model. Returns a confidence score from 0-100%.

WHEN TO USE:
- ALWAYS call this at the start of a session to get the baseline score
- Call this AFTER making changes to check if the score improved
- When the user asks 'what's my score?' or 'how good is my match?'

RETURNS: A JSON object with a 'score' field (0-100). If the model isn't loaded, returns score=-1 with an error message.""",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]
