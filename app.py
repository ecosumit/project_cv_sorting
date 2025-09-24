import streamlit as st
from pathlib import Path
#import ruamel
#import yaml as YAML
from ruamel.yaml import YAML
import pandas as pd
from src.ranker import rank_cvs_with_payload
from src.ui_components import show_results

st.set_page_config(page_title="CV Shortlister (Ollama)", layout="wide")
st.title("CV Shortlister")

with st.sidebar:
    st.markdown("### Config")
    cfg_file = st.file_uploader("Scoring Config (.yaml)", type=["yaml","yml"])
    if cfg_file:
        cfg = YAML(typ="safe").load(cfg_file.getvalue().decode("utf-8"))
    else:
        cfg = YAML(typ="safe").load(Path("config.yaml").read_text())

jd = st.file_uploader("Job Description (.txt)", type=["txt"])
cvs = st.file_uploader("CVs (.pdf/.docx/.txt)", type=["pdf","docx","txt"], accept_multiple_files=True)

col1, col2 = st.columns(2)
with col1:
    run = st.button("Score", use_container_width=True)
with col2:
    st.info("Use Ollama: `ollama pull llama3:8b` and `ollama pull nomic-embed-text` before running.", icon="ℹ️")

if run:
    if not jd or not cvs:
        st.error("Please upload a JD and at least one CV.")
        st.stop()
    jd_text = jd.getvalue().decode("utf-8")
    cv_paths = []
    upload_dir = Path("uploads"); upload_dir.mkdir(exist_ok=True)
    for f in cvs:
        p = upload_dir / f.name
        p.write_bytes(f.getvalue())
        cv_paths.append(p)
    with st.spinner("Scoring... (this may take a few minutes locally)"):
        df = rank_cvs_with_payload(cv_paths, jd_text, cfg)
        # Expand computed fields for table convenience
        df["kw"] = df["scores"].apply(lambda s: s["keywords"])
        df["sem"] = df["scores"].apply(lambda s: s["semantic"])
        df["rubric"] = df["scores"].apply(lambda s: s["rubric"])
        df["bonus_malus"] = df["scores"].apply(lambda s: s["bonus_malus"])
    st.success("Done")
    show_results(df)
