"""
Agent Orchestrator - Universal support with fixed tool schemas for Google Gemini.
"""
import os
import json
import asyncio
from typing import AsyncGenerator, Optional
from openai import OpenAI
import google.generativeai as genai

from .tools import (
    LatexDocument, TOOL_SCHEMAS,
    read_latex, edit_latex, edit_latex_section,
    extract_text_from_latex, score_resume
)

class AgentOrchestrator:
    """
    Orchestrates the agentic resume improvement loop.
    """
    def __init__(self, api_key: str, ml_model=None, ml_tokenizer=None):
        self.api_key = api_key.strip()
        self.provider = "openai" # default
        self.ml_model = ml_model
        self.ml_tokenizer = ml_tokenizer
        self.doc: Optional[LatexDocument] = None
        self.job_description: str = ""
        self.history = []

        if self.api_key.startswith("AIza"):
            self.provider = "google"
            genai.configure(api_key=self.api_key)
            # For Google, we use the manual declarations to avoid Pydantic schema errors
            self.model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
                system_instruction=SYSTEM_PROMPT,
                tools=self._get_google_tools()
            )
        elif self.api_key.startswith("sk-or-v1"):
            self.provider = "openrouter"
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
                default_headers={"HTTP-Referer": "http://localhost:3000", "X-Title": "JobFit Agent"}
            )
            self.model_name = "meta-llama/llama-3.1-8b-instruct:free"
        else:
            self.provider = "openai"
            self.client = OpenAI(api_key=self.api_key)
            self.model_name = "gpt-4o-mini"

        # Pre-format tools for OpenAI-compatible providers
        self.openai_tools = [
            {"type": "function", "function": {"name": s["name"], "description": s["description"], "parameters": s["parameters"]}}
            for s in TOOL_SCHEMAS
        ]

    def _get_google_tools(self):
        """Manually define tool declarations for Google to avoid Pydantic issues."""
        return [
            {
                "function_declarations": [
                    {
                        "name": s["name"],
                        "description": s["description"],
                        "parameters": s["parameters"]
                    } for s in TOOL_SCHEMAS
                ]
            }
        ]

    def start_session(self, latex_code: str, job_description: str):
        """Initialize a new improvement session."""
        self.doc = LatexDocument(latex_code)
        self.job_description = job_description
        
        if self.provider == "google":
            self.chat = self.model.start_chat(history=[])
        else:
            system_content = SYSTEM_PROMPT.replace("{{JD}}", job_description)
            self.history = [{"role": "system", "content": system_content}]

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a tool call and return the result."""
        try:
            if name == "read_latex":
                return read_latex(self.doc)
            elif name == "edit_latex":
                return edit_latex(self.doc, args.get("new_content", ""))
            elif name == "edit_latex_section":
                return edit_latex_section(self.doc, args.get("old_text", ""), args.get("new_text", ""))
            elif name == "score_resume":
                resume_text = extract_text_from_latex(self.doc.get_content())
                result = score_resume(resume_text, self.job_description, self.ml_model, self.ml_tokenizer)
                return json.dumps(result)
            return f"Unknown tool: {name}"
        except Exception as e:
            return f"Tool Execution Error: {str(e)}"

    async def process_message(self, user_message: str) -> AsyncGenerator[dict, None]:
        print(f"DEBUG: Received message from user: {user_message}")
        if not self.doc:
            yield {"type": "error", "content": "Session not initialized."}
            return
        
        yield {"type": "thinking", "content": "Agent is thinking..."}

        try:
            if self.provider == "google":
                async for chunk in self._process_google(user_message):
                    yield chunk
            else:
                async for chunk in self._process_openai_compatible(user_message):
                    yield chunk
        except Exception as e:
            print(f"DEBUG: CRITICAL ERROR in process_message: {str(e)}")
            yield {"type": "error", "content": f"System Error: {str(e)}"}
        
        yield {"type": "done"}

    async def _process_google(self, user_message: str) -> AsyncGenerator[dict, None]:
        """Process using Google Generative AI SDK (Async)."""
        response = await self.chat.send_message_async(user_message)
        
        for _ in range(12):
            if response.candidates[0].content.parts:
                tool_calls = [p.function_call for p in response.candidates[0].content.parts if p.function_call]
                text_parts = [p.text for p in response.candidates[0].content.parts if p.text]
                
                if text_parts:
                    yield {"type": "message", "content": " ".join(text_parts)}

                if not tool_calls:
                    break

                tool_responses = []
                for tc in tool_calls:
                    yield {"type": "tool_call", "tool": tc.name, "args": dict(tc.args)}
                    result = self._execute_tool(tc.name, dict(tc.args))
                    yield {"type": "tool_result", "tool": tc.name, "result": result[:300]}
                    
                    if tc.name in ("edit_latex", "edit_latex_section"):
                        yield {"type": "latex_update", "content": self.doc.get_content()}
                    
                    tool_responses.append(genai.types.Part.from_function_response(name=tc.name, response={"result": result}))
                
                response = await self.chat.send_message_async(tool_responses)
            else:
                break

    async def _process_openai_compatible(self, user_message: str) -> AsyncGenerator[dict, None]:
        """Process using OpenAI-compatible client."""
        self.history.append({"role": "user", "content": user_message})
        for _ in range(10):
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(None, lambda: self.client.chat.completions.create(
                model=self.model_name, messages=self.history, tools=self.openai_tools
            ))
            
            msg = resp.choices[0].message
            self.history.append(msg)
            
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    yield {"type": "tool_call", "tool": name, "args": args}
                    result = self._execute_tool(name, args)
                    yield {"type": "tool_result", "tool": name, "result": result[:300]}
                    if name in ("edit_latex", "edit_latex_section"):
                        yield {"type": "latex_update", "content": self.doc.get_content()}
                    self.history.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue
            else:
                yield {"type": "message", "content": msg.content or ""}
                break

SYSTEM_PROMPT = """You are JobFit Agent — a professional AI resume consultant. 

## Your Approach
- You are polite and reactive. If the user says "hi", greet them back.
- Do NOT start editing until the user asks you to start or gives a specific instruction.
- When asked to improve, use `read_latex` and `score_resume` first.
- Targeted edits only via `edit_latex_section`.
- Maintain valid LaTeX.
"""
