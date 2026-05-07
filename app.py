"""
app.py — Streamlit frontend for the RAG Codebase Q&A system.
"""

from __future__ import annotations

import json
import logging
import os
import time
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

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .answer-box {
        background: #0d1117;
        border-left: 4px solid #667eea;
        padding: 1.2rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.97rem;
        line-height: 1.7;
    }
    .stExpander { border: 1px solid #e2e8f0 !important; border-radius: 8px !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🔍 RAG Codebase Q&A")
    st.markdown("*Ask natural language questions about any GitHub repository.*")
    st.divider()

    st.markdown("### 📦 Index a Repository")
    repo_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/tiangolo/fastapi",
        help="Full GitHub repository URL.",
    )
    
    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        options=["ast", "recursive", "fixed", "semantic"],
        index=0,
        help=(
            "**AST** — split Python at function/class boundaries (best for code)\n\n"
            "**Recursive** — split by code structure characters\n\n"
            "**Fixed** — fixed 512-token sliding window\n\n"
            "**Semantic** — group by embedding similarity"
        ),
    )

    index_btn = st.button(
        "🚀 Index Repository", type="primary", use_container_width=True
    )

    if index_btn and repo_url:
        with st.spinner("Cloning and indexing… this may take a few minutes."):
            try:
                resp = requests.post(
                    f"{API_BASE}/index",
                    json={"repo_url": repo_url, "chunk_strategy": chunk_strategy},
                    timeout=600,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data["status"] == "already_indexed":
                        st.info("✅ Already indexed — ready to query.")
                    else:
                        st.success(
                            f"✅ Indexed **{data['num_chunks']:,}** chunks in "
                            f"{data['time_taken_s']:.1f}s "
                            f"(≈ ${data['estimated_cost_usd']:.4f})"
                        )
                    st.session_state["repo_name"] = data["repo_name"]
                    st.session_state["chunk_strategy"] = chunk_strategy
                    logger.info(
                        "Repository indexed via UI: repo=%s strategy=%s chunks=%s",
                        data["repo_name"],
                        chunk_strategy,
                        data["num_chunks"],
                    )
                else:
                    detail = resp.json().get("detail", "Unknown error")
                    logger.warning("Index request failed: status=%d detail=%s", resp.status_code, detail)
                    st.error(f"Indexing failed: {detail}")
            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to API backend")
                st.error(
                    "❌ Cannot connect to API. Is the backend running?\n\n"
                    "`uvicorn api.main:app --reload`"
                )
            except Exception as exc:
                logger.error("Unexpected error during indexing: %s", exc, exc_info=True)
                st.error(f"Unexpected error: {exc}")
    elif index_btn:
        st.warning("Please enter a GitHub URL first.")

    st.divider()

    if "repo_name" in st.session_state:
        st.markdown(f"**Active repo:** `{st.session_state['repo_name']}`")
        st.markdown(f"**Strategy:** `{st.session_state.get('chunk_strategy', 'ast')}`")
        st.divider()

    st.markdown("### ⚙️ Query Settings")
    alpha = st.slider(
        "Dense / Sparse Balance (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="α = 1.0 → pure semantic search.  α = 0.0 → pure BM25 keyword search.",
    )
    model_choice = st.selectbox(
        "LLM Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0
    )

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_qa, tab_eval = st.tabs(["💬 Q&A", "📊 Evaluation"])

# ─────────────────────── Q&A TAB ──────────────────────────────────────────────
with tab_qa:
    st.markdown('<div class="main-title">Ask Your Codebase</div>', unsafe_allow_html=True)
    st.caption(
        "Ask questions about any indexed GitHub repository. "
        "Answers are cited with exact source files and function names."
    )

    col_repo, col_strat = st.columns([3, 1])
    with col_repo:
        repo_override = st.text_input(
            "Repository name (auto-filled after indexing)",
            key="repo_name",
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

    question = st.text_area(
        "Your question",
        placeholder=(
            "How does FastAPI handle dependency injection?\n"
            "How is request validation implemented?\n"
            "Where is the routing logic defined?"
        ),
        height=100,
    )

    ask_btn = st.button("🔍 Ask", type="primary")

    if ask_btn and question and repo_override:
        with st.spinner("Retrieving context and generating answer…"):
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
                        st.metric("⏱️ Latency", f"{data['latency_ms']} ms")
                    with m2:
                        st.metric("🎯 Sources", len(data["sources"]))
                    with m3:
                        st.metric("🔤 Tokens", f"{data['tokens_used']:,}")
                    with m4:
                        st.metric("🤖 Model", data["model"])

                    st.markdown("### Answer")
                    st.markdown(
                        f'<div class="answer-box">{data["answer"]}</div>',
                        unsafe_allow_html=True,
                    )

                    if data["sources"]:
                        st.markdown("### 📁 Sources")
                        for i, src in enumerate(data["sources"]):
                            score_pct = (
                                f"{src['score']:.1%}" if src.get("score") else "N/A"
                            )
                            with st.expander(
                                f"[{i+1}] `{src['file_path']}` "
                                f"{'→ ' + src['function_name'] if src.get('function_name') else ''} "
                                f"(relevance: {score_pct})"
                            ):
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
                    logger.warning("Query failed: status=%d detail=%s", resp.status_code, detail)
                    st.error(f"Query failed: {detail}")

            except requests.exceptions.ConnectionError:
                logger.error("Cannot connect to API backend during query")
                st.error("❌ Cannot connect to API backend. Is it running?")
            except Exception as exc:
                logger.error("Unexpected error during query: %s", exc, exc_info=True)
                st.error(f"Unexpected error: {exc}")

    elif ask_btn and not repo_override:
        st.warning("Please index a repository first, or enter a repo name above.")
    elif ask_btn and not question:
        st.warning("Please enter a question.")

    with st.expander("💡 Example questions"):
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

# ─────────────────────── EVAL TAB ─────────────────────────────────────────────
with tab_eval:
    st.markdown("## 📊 Evaluation Results")
    st.markdown(
        "Comparison of 4 chunking strategies evaluated on 20 ground-truth Q&A pairs "
        "using the [RAGAS](https://docs.ragas.io/) framework."
    )

    ablation_path = Path("evaluation/results/ablation_results.json")
    ragas_path = Path("evaluation/results/ragas_scores.json")

    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_data = json.load(f)
        logger.debug("Ablation results loaded from disk")
    else:
        ablation_data = [
            {"strategy": "fixed",     "faithfulness": 0.78, "answer_relevancy": 0.74, "context_precision": 0.71, "context_recall": 0.69, "avg_answer_latency_ms": 950},
            {"strategy": "recursive", "faithfulness": 0.83, "answer_relevancy": 0.80, "context_precision": 0.77, "context_recall": 0.75, "avg_answer_latency_ms": 1020},
            {"strategy": "ast",       "faithfulness": 0.91, "answer_relevancy": 0.87, "context_precision": 0.84, "context_recall": 0.82, "avg_answer_latency_ms": 1150},
            {"strategy": "semantic",  "faithfulness": 0.86, "answer_relevancy": 0.83, "context_precision": 0.80, "context_recall": 0.78, "avg_answer_latency_ms": 1800},
        ]
        logger.info("No ablation results on disk — showing placeholder data")
        st.info(
            "📋 Showing sample data. "
            "Run `python evaluation/ablation_study.py --repo tiangolo_fastapi` "
            "to populate with real results."
        )

    df = pd.DataFrame(ablation_data)

    column_labels = {
        "strategy": "Strategy",
        "faithfulness": "Faithfulness ↑",
        "answer_relevancy": "Answer Relevancy ↑",
        "context_precision": "Context Precision ↑",
        "context_recall": "Context Recall ↑",
        "avg_answer_latency_ms": "Avg Latency (ms) ↓",
    }
    df_display = df[[c for c in column_labels if c in df.columns]].rename(
        columns=column_labels
    )

    st.markdown("### 📋 Strategy Comparison")

    def highlight_best(s: pd.Series) -> list[str]:
        is_best = s == (s.min() if "Latency" in s.name else s.max())
        return [
            "background-color: #d4edda; font-weight: bold" if v else "" for v in is_best
        ]

    try:
        styled = df_display.style.apply(highlight_best).format(
            {c: "{:.3f}" for c in df_display.columns if "Latency" not in c}
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    if "faithfulness" in df.columns:
        winner = df.loc[df["faithfulness"].idxmax()]
        st.success(
            f"🏆 **Winner: {winner['strategy'].upper()} chunking** — "
            f"Faithfulness: {winner['faithfulness']:.3f}"
        )

    st.markdown("### 📈 Metric Comparison")
    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]
    if metric_cols:
        st.bar_chart(df.set_index("strategy")[metric_cols], use_container_width=True)

    if "avg_answer_latency_ms" in df.columns:
        st.markdown("### ⏱️ Answer Generation Latency (ms)")
        st.bar_chart(
            df.set_index("strategy")[["avg_answer_latency_ms"]], use_container_width=True
        )

    if ragas_path.exists():
        st.markdown("### 🔬 Raw RAGAS Scores")
        with open(ragas_path) as f:
            st.json(json.load(f))