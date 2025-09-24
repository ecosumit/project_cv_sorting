from __future__ import annotations
from typing import Dict, Optional, List
import re
from pathlib import Path
import fitz
import docx

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-() ]{8,}\d")

def read_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    return "\n".join(page.get_text("text") for page in doc)

def read_docx(path: Path) -> str:
    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)

def read_text_file(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")

def read_cv(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return read_pdf(path)
    if ext == ".docx":
        return read_docx(path)
    return read_text_file(path)

def basic_fields(text: str) -> Dict[str, Optional[str]]:
    email = next(iter(EMAIL_RE.findall(text)), None)
    phone = next(iter(PHONE_RE.findall(text)), None)
    return {"email": email, "phone": phone}

def try_pyresparser(path: Path) -> Dict:
    """Attempt to use pyresparser if available; otherwise return empty dict."""
    try:
        from pyresparser import ResumeParser  # type: ignore
        data = ResumeParser(str(path)).get_extracted_data() or {}
        return data
    except Exception:
        return {}

def parse_resume(path: Path) -> Dict:
    text = read_cv(path)
    fields = basic_fields(text)
    pr = try_pyresparser(path)
    # Fallback enrichments
    skills = pr.get("skills") or []
    if not skills:
        # crude fallback skill detection
        skill_candidates = ["python","sql","pytorch","tensorflow","aws","gcp","azure","airflow","spark","kubernetes","docker"]
        t = text.lower()
        skills = [s for s in skill_candidates if s in t]
    years = 0.0
    years += len(re.findall(r"\b(\d{4})\s*-\s*(\d{4}|present|current)\b", text.lower())) * 1.5
    years += len(re.findall(r"(\d+(?:\.\d+)?)\s+years?", text.lower())) * 1.0
    years = min(years, 30.0)

    return {
        "raw_text": text,
        "email": fields["email"] or pr.get("email"),
        "phone": fields["phone"] or pr.get("mobile_number"),
        "name": pr.get("name"),
        "skills": skills,
        "total_experience_years": years if years else pr.get("total_experience"),
        "degree": pr.get("degree"),
        "designation": pr.get("designation"),
        "company_names": pr.get("company_names") or [],
        "meta": pr
    }
