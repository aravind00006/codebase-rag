
import json
import logging
import os
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Codebase Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE: str = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

# Colors use Streamlit CSS variables so they adapt to light and dark mode.
st.markdown(
    """
<style>
    .answer-box {
        border-left: 4px solid var(--primary-color, #667eea);
        padding: 1.2rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.97rem;
        line-height: 1.7;
    }
    .stExpander {
        border: 1px solid rgba(0,0,0,0.1) !important;
        border-radius: 8px !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## RAG Codebase Q&A")
    st.markdown("*Ask natural language questions about any GitHub repository.*")
    st.divider()

    st.markdown("### Index a Repository")
    repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/tiangolo/fastapi",
        help="Full GitHub repository URL.",
    )

    chunk_strategy = st.selectb



tab_qa, tab_eval = st.tabs(["Q&A", "Evaluation"])

with tab_qa:
    st.title("Ask Your Codebase")
    st.caption(
        "Ask questions about any indexed GitHub repository. "
        "Answers are cited with exact source files and function names."
    )

    col_repo, col_strat = st.columns([3, 1])
    with col_repo:
        repo_override = st.text_input(
            "Repository name (auto-filled after indexing)",
            value=st.session_state.get("repo_name", ""),
            placeholder="e.g. tiangolo_fastapi",
        )
    with col_strat:
        strategy_override = st.selectbox(
            "Strategy",
            ["ast", "recursive", "fixed", "semantic"],
            index=["ast", "recursive", "fixed", "semantic"].index(
                st.session_state.get("chunk_strategy", "ast")
            ),
        )

    # Pre-fill question from example buttons if one was clicked last run.
    default_question = st.session_state.pop("example_question", "")
    question = st.text_area(
        "Your question",
        value=default_question,
        placeholder=(
            "How does FastAPI handle dependency injection?\n"
            "How is request validation implemented?\n"
            "Where is the routing logic defined?"
        ),
        height=100,
    )

    ask_btn = st.button("Ask", type="primary")

    if ask_btn and question and repo_override:
        with st.spinner("Retrieving context and generating answer..."):
            logger.info(
                "Query submitted via UI: repo=%s strategy=%s preview='%s...'",
                repo_override,
                strategy_override,
                question[:60],
            )
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={
                        "question": question,
                        "repo_name": repo_override,
                        "chunk_strategy": strategy_override,
                        "top_k": 5,
                        "alpha": alpha,
                        "model": model_choice,
                    },
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    logger.info(
                        "Query response received: latency_ms=%d tokens=%d sources=%d",
                        data["latency_ms"],
                        data["tokens_used"],
                        len(data["sources"]),
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Latency", f"{data['latency_ms']} ms")
                    with m2:
                        st.metric("Sources", len(data["sources"]))
                    with m3:
                        st.metric("Tokens", f"{data['tokens_used']:,}")
                    with m4:
                        st.metric("Model", data["model"])

                    st.markdown("### Answer")
                   
                    with st.container(border=True):
                        st.markdown(data["answer"])

                    if data["sources"]:
                        st.markdown("### Sources")
                        for i, src in enumerate(data["sources"]):
                            score_pct = (
                                f"{src['score']:.1%}" if src.get("score") else "N/A"
                            )
                            fn = src.get("function_name", "")
                            label = (
                                f"[{i+1}] `{src['file_path']}`"
                                + (f" → {fn}" if fn else "")
                                + f" (relevance: {score_pct})"
                            )
                            with st.expander(label):
                                if src.get("start_line"):
                                    st.caption(
                                        f"Lines {src['start_line']}–{src['end_line']}"
                                    )
                                st.code(
                                    src["chunk_preview"],
                                    language=src.get("language", ""),
                                )
                else:
                    detail = resp.json().get("detail", "Unknown error")
                    logger.warning(
                        "Query failed: status=%d detail=%s", resp.status_code, detail
                    )
                    st.error(f"Query failed: {detail}")

            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to API backend during query")
                st.error("Cannot connect to API backend. Is it running?")
            except Exception as exc:
                logger.error(
                    "Unexpected error during query: %s", exc, exc_info=True
                )
                st.error(f"Unexpected error: {exc}")

    elif ask_btn and not repo_override:
        st.warning("Please index a repository first, or enter a repo name above.")
    elif ask_btn and not question:
        st.warning("Please enter a question.")

    st.markdown("#### Example questions")
    examples = [
        "How does FastAPI handle request validation?",
        "How is dependency injection implemented?",
        "Where is the routing logic defined?",
        "What does the serialize_response function do?",
        "How do I add custom middleware?",
        "How does FastAPI generate OpenAPI documentation?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:25]}"):
            st.session_state["example_question"] = ex
            st.rerun()