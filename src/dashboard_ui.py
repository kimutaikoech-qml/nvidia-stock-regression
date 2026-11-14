"""Shared visual language for the Streamlit financial workspace."""
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

GREEN = "#76B900"
COLORS = [GREEN, "#63B3ED", "#E5BD65", "#B69AF7"]


def apply_style():
    st.html((Path(__file__).with_name("dashboard.css")).read_text())


def header(eyebrow, title, subtitle):
    st.caption(eyebrow.upper())
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
