from __future__ import annotations
import streamlit as st
import pandas as pd

def show_results(df: pd.DataFrame):
    st.subheader("Ranking")
    st.dataframe(df[["file","name","email","gate_pass","total_score","kw","sem","rubric","bonus_malus"]] if set(["kw","sem","rubric","bonus_malus"]).issubset(df.columns) else df, use_container_width=True)

    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "ranking.csv", "text/csv")
    # Per-candidate expandable details
    for _, row in df.iterrows():
        with st.expander(f"Details • {row.get('file')} • score={row.get('total_score'):.3f}"):
            st.write("**Gate pass:**", row.get("gate_pass"))
            if row.get("gate_reasons"):
                st.write("**Gate reasons:**", row.get("gate_reasons"))
            st.write("**Rubric details:**")
            st.json(row.get("rubric_details"))
            st.write("**Explanation:**")
            st.write(row.get("overall_comment"))
