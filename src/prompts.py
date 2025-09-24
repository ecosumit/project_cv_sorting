from langchain.prompts import PromptTemplate

RUBRIC_JSON_PROMPT = PromptTemplate.from_template(
    """You are a meticulous technical recruiter.

Task: Score this candidate on the rubric below for the given Job Description (JD).
Return strict JSON with fields:
{{ "rubric": [{{ "name": str, "score": int (0-5), "reason": str }}], "overall_comment": str }}

Guidelines:
- Use ONLY evidence from the CV text provided.
- Be conservative: if not evidenced, score lower.
- No personal attributes not present in CV.
- Keep reasons concise (<= 2 sentences each).

JD:
<<<{jd}>>>

CV:
<<<{cv}>>>

Rubric:
{rubric}
"""
)
