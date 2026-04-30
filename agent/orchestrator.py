"""
Agent Orchestrator - Agentic loop using Gemini Function Calling.

The LLM receives the user's message, the tool schemas, and decides which tools
to call. Results stream back to the frontend via WebSocket.
"""
import os
import json
import abc
from typing import AsyncGenerator, Optional
import google.generativeai as genai
from google.generativeai.types import content_types

from .tools import (
    LatexDocument, TOOL_SCHEMAS,
    read_latex, edit_latex, edit_latex_section,
    extract_text_from_latex, score_resume
)


class AgentOrchestrator:
    """
    Orchestrates the agentic resume improvement loop.
    Uses Gemini function calling so the LLM decides which tools to invoke.
    """
    def __init__(self, api_key: str, ml_model=None, ml_tokenizer=None):
        genai.configure(api_key=api_key)
        
        # Convert our tool schemas to Gemini-compatible format
        self.tools = genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name=s["name"],
                    description=s["description"],
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            k: genai.protos.Schema(type=genai.protos.Type.STRING, description=v.get("description", ""))
                            for k, v in s["parameters"].get("properties", {}).items()
                        },
                        required=s["parameters"].get("required", [])
                    )
                )
                for s in TOOL_SCHEMAS
            ]
        )
        
        self.model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            tools=[self.tools],
            system_instruction=SYSTEM_PROMPT,
        )
        
        self.ml_model = ml_model
        self.ml_tokenizer = ml_tokenizer
        self.doc: Optional[LatexDocument] = None
        self.job_description: str = ""
        self.chat = None  # Will hold a multi-turn chat session
    
    def start_session(self, latex_code: str, job_description: str):
        """Initialize a new improvement session with the user's resume and JD."""
        self.doc = LatexDocument(latex_code)
        self.job_description = job_description
        
        # Prime the chat with context about this session so it persists in history
        self.chat = self.model.start_chat(
            enable_automatic_function_calling=False,
            history=[
                {"role": "user", "parts": [f"""I need help improving my LaTeX resume for a specific job. Here is the job description I'm targeting:\n\n{job_description}\n\nI have uploaded my LaTeX resume. Use the read_latex tool to see it, and score_resume to get the baseline score. Then help me improve it based on my instructions."""]},
                {"role": "model", "parts": ["I understand! I have the job description. I'll use my tools to read your resume, score it, and then help you improve it. What would you like me to do?"]}
            ]
        )
    
    def _execute_tool(self, function_call) -> str:
        """Execute a tool call from the LLM and return the result."""
        name = function_call.name
        args = {k: v for k, v in function_call.args.items()}
        
        if name == "read_latex":
            return read_latex(self.doc)
        elif name == "edit_latex":
            return edit_latex(self.doc, args["new_content"])
        elif name == "edit_latex_section":
            return edit_latex_section(self.doc, args["old_text"], args["new_text"])
        elif name == "score_resume":
            resume_text = extract_text_from_latex(self.doc.get_content())
            result = score_resume(resume_text, self.job_description, self.ml_model, self.ml_tokenizer)
            return json.dumps(result)
        else:
            return f"Unknown tool: {name}"
    
    async def process_message(self, user_message: str) -> AsyncGenerator[dict, None]:
        """
        Process a user message through the agentic loop.
        Yields events that the WebSocket sends to the frontend:
          - {"type": "thinking", "content": "..."}
          - {"type": "tool_call", "tool": "...", "args": {...}}
          - {"type": "tool_result", "tool": "...", "result": "..."}
          - {"type": "latex_update", "content": "..."} 
          - {"type": "message", "content": "..."}
          - {"type": "done"}
        """
        if not self.chat or not self.doc:
            yield {"type": "error", "content": "Session not initialized. Send latex_code and job_description first."}
            return
        
        yield {"type": "thinking", "content": "Processing your request..."}
        
        # Send user message to Gemini (chat history automatically accumulates)
        response = self.chat.send_message(user_message)
        
        # Agentic loop: keep going while the model wants to call tools
        max_iterations = 15  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Check if the model wants to call a function
            part = response.candidates[0].content.parts[0]
            
            if hasattr(part, 'function_call') and part.function_call.name:
                fc = part.function_call
                
                # Notify frontend about the tool call
                yield {
                    "type": "tool_call",
                    "tool": fc.name,
                    "args": dict(fc.args)
                }
                
                # Execute the tool
                result = self._execute_tool(fc)
                
                # Notify frontend about the result
                yield {
                    "type": "tool_result",
                    "tool": fc.name,
                    "result": result[:500] if len(result) > 500 else result  # Truncate for display
                }
                
                # If a latex edit was made, send the updated content to frontend
                if fc.name in ("edit_latex", "edit_latex_section"):
                    yield {
                        "type": "latex_update",
                        "content": self.doc.get_content()
                    }
                
                # Send the function result back to Gemini so it can continue
                response = self.chat.send_message(
                    genai.protos.Content(
                        parts=[genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fc.name,
                                response={"result": result}
                            )
                        )]
                    )
                )
            else:
                # Model returned a text response (done with tools)
                text = part.text if hasattr(part, 'text') else str(part)
                yield {"type": "message", "content": text}
                break
        
        yield {"type": "done"}


# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are JobFit Agent — an expert AI resume optimizer. You help users improve their LaTeX resumes to better match specific job descriptions.

## Your Capabilities
You have access to these tools:
1. **read_latex** — Read the current LaTeX resume source code
2. **edit_latex_section** — Make targeted edits by replacing specific text sections
3. **edit_latex** — Rewrite the entire document (use sparingly)
4. **score_resume** — Score the current resume against the job description (0-100%)

## Your Workflow
When asked to improve a resume:
1. First, use `read_latex` to understand the current resume
2. Use `score_resume` to get the baseline score
3. Analyze the gap between current score and target
4. Make targeted edits using `edit_latex_section` to improve keyword alignment, bullet points, and skills
5. After each significant change, use `score_resume` to check progress
6. Continue until the target score is reached or no more improvements can be made

## Rules
- NEVER fabricate experiences, degrees, certifications, or skills that aren't in the original resume
- You CAN rephrase, reorganize, and optimize wording
- You CAN reorder skills to prioritize those matching the JD
- You CAN improve bullet points to use the STAR framework with quantifiable metrics (if the data exists)
- You CAN add keywords from the JD into the summary/objective section IF they relate to existing skills
- Always maintain valid LaTeX syntax
- Explain each change you make and why
"""
