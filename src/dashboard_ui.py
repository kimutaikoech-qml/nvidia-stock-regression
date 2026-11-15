"""Shared visual language for the Streamlit financial workspace."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

GREEN = "#76B900"
COLORS = [GREEN, "#63B3ED", "#E5BD65", "#B69AF7"]
LOGO_PATH = Path(__file__).resolve().parents[1] / "assets" / "nvidia-logo.png"


def apply_style():
    st.html((Path(__file__).with_name("dashboard.css")).read_text())


def sidebar_logo():
    # The asset's original black canvas provides contrast and built-in clear space.
    with st.container(key="nvidia_sidebar_logo"):
        st.image(str(LOGO_PATH), width=240)


def header(eyebrow, title, subtitle, logo=False):
    st.caption(eyebrow.upper())
    if logo:
        with st.container(key="nvidia_heading"):
            mark, heading = st.columns([1, 4], vertical_alignment="center")
            with mark:
                st.image(str(LOGO_PATH), width=160)
            with heading:
                st.title(title)
    else:
        st.title(title)
    st.write(subtitle)


def chart(fig, key, percent=False, height=360):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", colorway=COLORS,
        font=dict(family="Arial, sans-serif", color="#AAB6C5", size=12),
        margin=dict(l=16, r=16, t=24, b=16), height=height,
        hovermode="x unified", legend=dict(orientation="h", y=1.12, title_text=""),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#252E3A", zerolinecolor="#35404D")
    if percent:
        fig.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displaylogo": False, "responsive": True}, key=key)


def pending(title, description):
    with st.container(border=True):
        st.caption("AVAILABLE AFTER TRAINING")
        st.subheader(title)
        st.write(description)


def empty_prediction_chart():
    fig = go.Figure()
    fig.add_annotation(text="Predicted vs actual prices will appear after training",
                       x=.5, y=.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=15, color="#AAB6C5"))
    fig.update_xaxes(title="Trading date", visible=False)
    fig.update_yaxes(title="Adjusted price (USD)", visible=False)
    chart(fig, "prediction_placeholder", height=300)
