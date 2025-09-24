from __future__ import annotations
from typing import Dict, Any, List, Tuple
import re
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from .prompts import RUBRIC_JSON_PROMPT

def chunk_text(t: str, max_chars: int = 1500, overlap: int = 150) -> List[str]:
    t = re.sub(r"\s+", " ", t).strip()
    chunks, i = [], 0
    while i < len(t):
        chunks.append(t[i:i+max_chars])
        i += max_chars - overlap
    return chunks

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)

def semantic_score(jd: str, cv_text: str, emb_model: str = "nomic-embed-text", topk: int = 8) -> float:
    embedder = OllamaEmbeddings(model=emb_model)
    cv_chunks = chunk_text(cv_text)
    vecs = embedder.embed_documents([jd] + cv_chunks)
    jd_vec = np.array(vecs[0], dtype=np.float32)
    sims = []
    for v in vecs[1:]:
        sims.append(cosine(jd_vec, np.array(v, dtype=np.float32)))
    top = sorted(sims, reverse=True)[:topk] or [0.0]
    return 0.6 * max(top) + 0.4 * (sum(top)/len(top))

def keyword_score(cv_text: str, critical: Dict[str,int], desired: Dict[str,int]) -> Tuple[float, Dict[str,int]]:
    text = cv_text.lower()
    score = 0
    hitmap: Dict[str,int] = {}
    for k,w in (critical or {}).items():
        if re.search(rf"\b{re.escape(k.lower())}\b", text):
            score += 2*w; hitmap[k] = 2*w
    for k,w in (desired or {}).items():
        if re.search(rf"\b{re.escape(k.lower())}\b", text):
            score += 1*w; hitmap[k] = hitmap.get(k,0) + 1*w
    max_possible = 2*sum((critical or {}).values()) + sum((desired or {}).values())
    return (score / max(1, max_possible)), hitmap

def must_have_gate(cv_text: str, cfg: Dict[str,Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    yrs = 0.0
    yrs += len(re.findall(r"\b(\d{4})\s*-\s*(\d{4}|present|current)\b", cv_text.lower())) * 1.5
    yrs += len(re.findall(r"(\d+(?:\.\d+)?)\s+years?", cv_text.lower())) * 1.0
    if yrs < cfg.get("min_years_total", 0):
        reasons.append(f"Years of experience {yrs:.1f} < {cfg.get('min_years_total',0)}")
    for req in cfg.get("required_skills", []):
        if not re.search(rf"\b{re.escape(req.lower())}\b", cv_text.lower()):
            reasons.append(f"Missing required skill: {req}")
    return len(reasons) == 0, reasons

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def rubric_eval(jd: str, cv_text: str, rubric_items: List[Dict[str,str]], model: str = "llama3:8b") -> Tuple[float, List[Dict[str,Any]], str]:
    rubric_str = "\n".join([f"- {it['name']}: {it['question']}" for it in rubric_items])
    prompt = RUBRIC_JSON_PROMPT.format(jd=jd, cv=cv_text[:20000], rubric=rubric_str)
    llm = ChatOllama(model=model, temperature=0.1)
    out = llm.invoke(prompt)
    content = out.content.strip()
    # crude JSON recovery
    start, end = content.find("{"), content.rfind("}")
    if start == -1 or end == -1:
        return 0.0, [], "Parser failed"
    import json
    data = json.loads(content[start:end+1])
    items = data.get("rubric", [])
    scores = [max(0, min(5, int(it.get("score",0))))/5 for it in items if isinstance(it, dict)]
    return (float(np.mean(scores)) if scores else 0.0), items, data.get("overall_comment","")

def bonus_malus(cv_text: str, cfg: Dict[str,Any]) -> float:
    bonus = 0.0
    if re.search(r"\b(lead|manager|head|principal)\b", cv_text.lower()):
        bonus += cfg.get("leadership_bonus", 0.0)
    if re.search(r"\b(arxiv|publication|paper|peer-reviewed)\b", cv_text.lower()):
        bonus += cfg.get("publication_bonus", 0.0)
    if len(re.findall(r"\b(month|year)\b", cv_text.lower())) > 40:
        bonus += cfg.get("job_hop_malus", 0.0)
    return bonus
