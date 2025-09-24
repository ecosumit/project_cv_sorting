from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
#import yaml as YAML
from ruamel.yaml import YAML
from .parsing import parse_resume, read_cv
from .scoring import must_have_gate, keyword_score, semantic_score, rubric_eval, bonus_malus

def score_one(cv_path: Path, jd_text: str, cfg: Dict[str,Any]) -> Dict[str,Any]:
    parsed = parse_resume(cv_path)
    gate_ok, gate_reasons = must_have_gate(parsed["raw_text"], cfg["must_haves"])
    kw_score, hits = keyword_score(parsed["raw_text"], cfg["keywords"]["critical"], cfg["keywords"]["desired"])
    sem = semantic_score(jd_text, parsed["raw_text"], emb_model=cfg["models"]["embeddings"], topk=cfg["semantic"]["topk"])
    r_score, r_items, r_comment = rubric_eval(jd_text, parsed["raw_text"], cfg["rubric"], model=cfg["models"]["llm"])
    bm = bonus_malus(parsed["raw_text"], cfg["bonus_malus"])
    total = (
        cfg["weights"]["keywords"] * kw_score +
        cfg["weights"]["semantic"] * sem +
        cfg["weights"]["rubric"]   * r_score +
        cfg["weights"]["bonus_malus"] * (0.5 + bm/2)
    )
    return {
        "file": cv_path.name,
        "name": parsed.get("name"),
        "email": parsed.get("email"),
        "phone": parsed.get("phone"),
        "gate_pass": gate_ok,
        "gate_reasons": gate_reasons,
        "scores": {"keywords": kw_score, "semantic": sem, "rubric": r_score, "bonus_malus": bm},
        "hits": hits,
        "rubric_details": r_items,
        "overall_comment": r_comment,
        "total_score": float(total),
    }

def rank_cvs(cv_paths: List[Path], jd_path: Path, cfg_path: Path) -> pd.DataFrame:
    yaml = YAML(typ="safe")
    cfg = yaml.load(Path(cfg_path).read_text())
    jd = Path(jd_path).read_text(encoding="utf-8")
    rows = [score_one(p, jd, cfg) for p in cv_paths]
    df = pd.DataFrame(rows).sort_values(["gate_pass", "total_score"], ascending=[False, False])
    return df

def rank_cvs_with_payload(cv_paths: List[Path], jd_text: str, cfg: Dict[str,Any]) -> pd.DataFrame:
    rows = [score_one(p, jd_text, cfg) for p in cv_paths]
    df = pd.DataFrame(rows).sort_values(["gate_pass", "total_score"], ascending=[False, False])
    return df
