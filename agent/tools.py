import re
import asyncio

def extract_text_from_latex(latex_code: str) -> str:
    """
    Very basic heuristic to strip common LaTeX tags to get readable text for scoring.
    For a production system, use something like pylatexenc.
    """
    # Remove comments
    text = re.sub(r'%.*?\n', '\n', latex_code)
    # Remove commands like \textbf{...} -> ...
    text = re.sub(r'\\[a-zA-Z]+\*?(?:\[.*?\])?\{([^{}]*)\}', r'\1', text)
    # Remove environment begin/end tags
    text = re.sub(r'\\begin\{.*?\}', '', text)
    text = re.sub(r'\\end\{.*?\}', '', text)
    # Remove formatting commands
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_gaps(current_score: float, target_score: float) -> str:
    """
    In a full implementation using a multi-headed model, this would look at micro/macro scores.
    Since we are using a single regression head right now, we use a generic gap analysis prompt.
    """
    gap = target_score - current_score
    if gap > 20:
        return "Major overhaul needed. Focus on deeply integrating keywords from the job description into the professional summary and rewriting experience bullets using the STAR framework with quantifiable metrics."
    elif gap > 10:
        return "Moderate improvements needed. Ensure all hard skills from the job description are explicitly listed and that experience bullet points directly reflect the required responsibilities."
    else:
        return "Minor tweaks needed. Reorder skills to match the job description priority and refine the wording in the most recent experience section."

async def verify_factuality(llm_provider, original_text: str, modified_latex: str) -> bool:
    """
    Uses the LLM to verify that no new skills, degrees, or false experiences were hallucinated.
    """
    modified_text = extract_text_from_latex(modified_latex)
    
    prompt = f"""
    Compare the ORIGINAL resume text with the MODIFIED resume text.
    Your task is to determine if the modified resume contains any hallucinated facts (e.g., new skills, new degrees, new job titles, or inflated years of experience) that are NOT supported by the original text.
    Rephrasing and reorganizing is allowed. Inventing facts is NOT allowed.
    
    ORIGINAL:
    {original_text}
    
    MODIFIED:
    {modified_text}
    
    Respond with ONLY 'PASS' if no facts were fabricated, or 'FAIL' if new facts were invented.
    """
    
    response = await llm_provider.generate(prompt=prompt, system_prompt="You are a strict fact-checker.")
    return 'PASS' in response.upper()
