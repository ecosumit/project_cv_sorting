from __future__ import annotations
from typing import Dict, Any, List
from langchain_community.chat_models import ChatOllama
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
import json

class JDParsed(BaseModel):
    title: str
    required_skills: List[str]
    nice_to_have: List[str]
    responsibilities: List[str]
    min_years_total: int

def parse_jd(jd_text: str, model: str = "llama3:8b") -> JDParsed:
    response_schemas = [
        ResponseSchema(name="title", description="Job title string"),
        ResponseSchema(name="required_skills", description="List of required skills"),
        ResponseSchema(name="nice_to_have", description="List of good-to-have skills"),
        ResponseSchema(name="responsibilities", description="List of responsibilities"),
        ResponseSchema(name="min_years_total", description="Minimum total years of experience (int)"),
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You extract structured fields from a Job Description."),
        ("human", "JD:\n{jd}\n\nReturn JSON with fields as specified.\n{format}")
    ])

    llm = ChatOllama(model=model, temperature=0.1)
    chain = prompt | llm
    out = chain.invoke({"jd": jd_text, "format": format_instructions})
    try:
        data = parser.parse(out.content)
    except Exception:
        # fall back to best-effort JSON parse
        try:
            data = json.loads(out.content)
        except Exception:
            data = {
                "title": "",
                "required_skills": [],
                "nice_to_have": [],
                "responsibilities": [],
                "min_years_total": 0
            }
    return JDParsed(**data)
