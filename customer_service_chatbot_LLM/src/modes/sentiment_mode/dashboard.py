import streamlit as st
from datetime import datetime
from typing import List, Dict, Any


def _normalize_history(history: List[Dict[str, Any]]):
    # Ensure minimal structure
    rows = []
    for m in history:
        label = None
        score = None
        ts = m.get("timestamp")
        if isinstance(ts, datetime):
            ts_val = ts
        else:
            ts_val = None

        s = m.get("sentiment") or {}
        if isinstance(s, dict):
            label = s.get("label")
            score = s.get("score")

        rows.append({"timestamp": ts_val, "label": label, "score": score})
    return rows


def render_sentiment_dashboard(history: List[Dict[str, Any]] = None):
    """Render a small dashboard with sentiment graphs.

    Uses `st.session_state.chat_history` by default when `history` is None.
    """
    if history is None:
        history = st.session_state.get("chat_history", [])

    st.markdown("# 📊 Sentiment Dashboard")

    if not history:
        st.info("No chat history available. Interact with the assistant and enable sentiment analysis to populate this dashboard.")
        return

    rows = _normalize_history(history)

    # Aggregate counts
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    scores = []
    times = []
    for r in rows:
        lbl = (r.get("label") or "neutral").lower()
        if lbl not in counts:
            counts[lbl] = 0
        counts[lbl] += 1
        sc = r.get("score")
        if sc is not None:
            try:
                scores.append(float(sc))
            except Exception:
                pass
        ts = r.get("timestamp")
        times.append({"timestamp": ts, "score": sc, "label": lbl})

    # Top metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Messages", len(rows))
    with col2:
        st.metric("Positive", counts.get("positive", 0))
    with col3:
        st.metric("Negative", counts.get("negative", 0))

    # Bar chart for distribution
    try:
        import pandas as pd

        dist_df = pd.DataFrame([counts])
        st.markdown("### Sentiment Distribution")
        st.bar_chart(dist_df.T)

        # Score trend over time
        st.markdown("### Sentiment Score Over Time")
        times_df = pd.DataFrame(times)
        # If timestamps exist, use them for index
        if "timestamp" in times_df.columns and times_df["timestamp"].notnull().any():
            times_df = times_df.sort_values("timestamp")
            times_df = times_df.set_index("timestamp")
        st.line_chart(times_df["score"])

        # Simple table
        st.markdown("### Recent Messages (sentiment)")
        preview = []
        for i, m in enumerate(history[::-1][:10], 1):
            preview.append({"when": m.get("timestamp"), "mode": m.get("mode"), "label": (m.get("sentiment") or {}).get("label"), "score": (m.get("sentiment") or {}).get("score"), "question": (m.get("question") or "")[:120]})
        st.table(pd.DataFrame(preview))

    except Exception:
        # Minimal non-pandas fallback
        st.markdown("### Sentiment Distribution")
        st.write(counts)
        st.markdown("### Scores")
        st.write(scores[:50])
