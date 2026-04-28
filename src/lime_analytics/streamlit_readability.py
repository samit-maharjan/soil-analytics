"""Larger, more readable type + spacing for Streamlit (called from each app page)."""

from __future__ import annotations

import streamlit as st


def inject_readability_css(emphasize_radio: bool = False) -> None:
    """
    Bump base text size in the main area. If `emphasize_radio`, add extra line spacing for
    `st.radio` (used for FESEM options).
    """
    extra = ""
    if emphasize_radio:
        extra = """
        /* FESEM wizard: larger, roomier options */
        section.main [data-testid="stRadio"] [data-baseweb="radio"] > div,
        section.main div[role="radiogroup"] > * {
            margin-bottom: 0.55rem !important;
        }
        section.main [data-testid="stRadio"] label p,
        section.main [data-baseweb="radio"] label p,
        section.main div[role="radiogroup"] label p {
            line-height: 1.75 !important;
            font-size: 1.1rem !important;
            margin-top: 0.4rem !important;
            margin-bottom: 0.4rem !important;
            padding: 0.2rem 0 0.35rem 0 !important;
        }
        """
    st.markdown(
        f"""
        <style>
        .block-container {{ max-width: 100%; font-size: 1.05rem; }}
        section.main p, section.main li,
        [data-testid="stMainBlockContainer"] p
            {{ font-size: 1.1rem !important; line-height: 1.62 !important; }}
        section.main h1 {{ font-size: 2.1rem !important; line-height: 1.22 !important; }}
        section.main h2 {{ font-size: 1.55rem !important; line-height: 1.3 !important; }}
        section.main h3, section.main h4 {{ font-size: 1.28rem !important; line-height: 1.4 !important; margin-top: 0.6rem; }}
        [data-testid="stCaptionContainer"] p {{ font-size: 1rem !important; }}
        /* Dataframe: no horizontal scroll; wrap long text */
        section.main [data-testid="stDataFrame"] {{
            font-size: 0.95rem;
            overflow-x: hidden !important;
            max-width: 100% !important;
        }}
        section.main [data-testid="stDataFrame"] [role="gridcell"],
        section.main [data-testid="stDataFrame"] [role="cell"] {{
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }}
        section.main [data-testid="stDataFrame"] [class*="dvn-"] p,
        section.main [data-testid="stDataFrame"] [class*="dvn-"] div {{
            white-space: normal !important;
            word-wrap: break-word !important;
        }}
        /* HTML inference tables (see streamlit_tables.py) — real text wrap, not Glide clip */
        section.main .sa-inf-table-wrap {{ width: 100%; max-width: 100%; overflow-x: hidden; }}
        section.main table.sa-inf-table {{
            width: 100%;
            table-layout: fixed;
            border-collapse: collapse;
            font-size: 0.95rem;
            margin: 0.35rem 0 0.9rem 0;
        }}
        section.main table.sa-inf-table th.sa-inf-th,
        section.main table.sa-inf-table td.sa-inf-td {{
            border: 1px solid rgba(128, 128, 128, 0.4);
            padding: 0.5rem 0.65rem;
            vertical-align: top;
            text-align: left;
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: anywhere !important;
            word-break: break-word;
            min-width: 0;
            line-height: 1.45;
        }}
        section.main table.sa-inf-table th.sa-inf-th {{
            font-weight: 600;
            background: rgba(128, 128, 128, 0.14);
        }}
        .stAlert p, [data-baseweb="notification"] p
            {{ font-size: 1.08rem !important; line-height: 1.6 !important; }}
        {extra}
        </style>
        """,
        unsafe_allow_html=True,
    )