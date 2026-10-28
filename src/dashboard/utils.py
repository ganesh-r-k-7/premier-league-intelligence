"""Utility functions for the Premier League Intelligence Dashboard."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Color palette for consistent styling
COLORS = {
    "primary": "#1a5c9e",       # Deep blue
    "secondary": "#2d8f47",     # Green
    "accent": "#e87722",        # Orange
    "warning": "#d62828",       # Red
    "background": "#f8f9fa",    # Light gray
    "text": "#212529",          # Dark gray
    "muted": "#6c757d",         # Medium gray
    "home": "#1a5c9e",          # Home team blue
    "away": "#d62828",          # Away team red
    "draw": "#6c757d",          # Draw gray
}

POSITION_BAND_COLORS = {
    "GK": "#1a5c9e",
    "DEF": "#2d8f47",
    "MID": "#e87722",
    "WING": "#9b59b6",
    "FWD": "#d62828",
}


def apply_custom_css() -> None:
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Main container */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        /* Headers */
        h1, h2, h3 {
            color: #1a5c9e;
            font-weight: 600;
        }

        h1 {
            border-bottom: 3px solid #1a5c9e;
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }

        h2 {
            margin-top: 2rem;
            margin-bottom: 1rem;
        }

        h3 {
            margin-top: 1.5rem;
            margin-bottom: 0.75rem;
        }

        /* Metric cards */
        [data-testid="metric-container"] {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        /* Sidebar */
        .css-1d391kg {
            background: #f8f9fa;
        }

        /* Dataframes */
        .stDataFrame {
            border: 1px solid #dee2e6;
            border-radius: 8px;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s;
        }

        .stButton > button[kind="primary"] {
            background: #1a5c9e;
            border-color: #1a5c9e;
        }

        .stButton > button[kind="primary"]:hover {
            background: #154a80;
            border-color: #154a80;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px 6px 0 0;
            padding: 0.5rem 1rem;
        }

        /* Alert boxes */
        .stAlert {
            border-radius: 8px;
        }

        /* Progress bar */
        .stProgress > div > div {
            border-radius: 4px;
        }

        /* Expanders */
        .streamlit-expanderHeader {
            font-weight: 500;
            color: #1a5c9e;
        }

        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_number(value: float | int, decimals: int = 1) -> str:
    """Format a number for display.

    Args:
        value: Number to format.
        decimals: Number of decimal places.

    Returns:
        Formatted string.
    """
    if pd.isna(value):
        return "—"
    if isinstance(value, float):
        return f"{value:,.{decimals}f}"
    return f"{value:,}"


def format_percentile(value: float) -> str:
    """Format a percentile value.

    Args:
        value: Percentile (0-100).

    Returns:
        Formatted string with percentile symbol.
    """
    if pd.isna(value):
        return "—"
    return f"{value:.0f}th"


def create_percentile_bar(value: float, color: str = "#1a5c9e") -> go.Figure:
    """Create a horizontal bar chart for a percentile.

    Args:
        value: Percentile value (0-100).
        color: Bar color.

    Returns:
        Plotly figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[value],
        y=[""],
        orientation="h",
        marker_color=color,
        marker_line_width=0,
        width=0.5,
        hovertemplate=f"Percentile: {value:.0f}th<extra></extra>",
    ))

    # Add reference line at 50th percentile
    fig.add_vline(x=50, line_dash="dash", line_color="#6c757d", line_width=1)

    fig.update_layout(
        xaxis=dict(range=[0, 100], showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        height=40,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    return fig


def create_radar_chart(
    df: pd.DataFrame,
    player_col: str = "player_name",
    metrics: list[str] = None,
    title: str = "",
) -> go.Figure:
    """Create a radar/spider chart for player comparison.

    Args:
        df: DataFrame with player data.
        player_col: Column with player names.
        metrics: List of metric columns to plot.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    if metrics is None:
        metrics = ["goals_per_90", "assists_per_90", "xg_per_90", "shots_per_90", "xa_per_90"]

    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        return go.Figure()

    fig = go.Figure()

    for _, row in df.iterrows():
        values = [row[m] for m in available_metrics]
        # Normalize to 0-100 scale for radar
        normalized = []
        for i, m in enumerate(available_metrics):
            col_data = df[m].dropna()
            if len(col_data) > 0 and col_data.max() > col_data.min():
                norm = (values[i] - col_data.min()) / (col_data.max() - col_data.min()) * 100
            else:
                norm = 50
            normalized.append(norm)

        # Close the loop
        normalized.append(normalized[0])
        labels = [m.replace("_", " ").title() for m in available_metrics]
        labels.append(labels[0])

        fig.add_trace(go.Scatterpolar(
            r=normalized,
            theta=labels,
            fill="toself",
            name=row[player_col],
            line=dict(width=2),
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
        ),
        showlegend=True,
        title=title,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def create_bar_comparison(
    df: pd.DataFrame,
    metric: str,
    player_col: str = "player_name",
    team_col: str = "team_name",
    color_col: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    """Create a horizontal bar chart for player comparison.

    Args:
        df: DataFrame with player data.
        metric: Metric column to plot.
        player_col: Player name column.
        team_col: Team name column.
        color_col: Optional column to color by.
        title: Chart title.

    Returns:
        Plotly figure.
    """
    if metric not in df.columns:
        return go.Figure()

    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return go.Figure()

    plot_df = plot_df.sort_values(metric, ascending=True)

    if color_col and color_col in plot_df.columns:
        colors = plot_df[color_col].map(POSITION_BAND_COLORS).fillna(COLORS["primary"])
    else:
        colors = COLORS["primary"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=plot_df[player_col] + " (" + plot_df[team_col] + ")",
        x=plot_df[metric],
        orientation="h",
        marker_color=colors,
        text=plot_df[metric].apply(lambda x: f"{x:.2f}"),
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title=metric.replace("_", " ").title(),
        yaxis_title="",
        height=max(200, len(plot_df) * 40 + 100),
        margin=dict(l=200, r=50, t=50, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def show_data_warning(message: str) -> None:
    """Show a styled warning message about data limitations.

    Args:
        message: Warning message.
    """
    st.warning(
        f"⚠️ **Data Limitation**: {message}\n\n"
        "This dashboard uses Premier League 2015/16 season data only (StatsBomb Open Data). "
        "Results are historical analysis, not current predictions."
    )


def show_info_box(title: str, content: str) -> None:
    """Show an info box with title and content.

    Args:
        title: Box title.
        content: Box content (markdown supported).
    """
    st.markdown(
        f"""
        <div style="
            background: #e7f3ff;
            border-left: 4px solid #1a5c9e;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        ">
            <strong style="color: #1a5c9e;">{title}</strong><br>
            {content}
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_empty_state(message: str, icon: str = "🔍") -> None:
    """Show an empty state message.

    Args:
        message: Message to display.
        icon: Emoji icon.
    """
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: 3rem;
            color: #6c757d;
        ">
            <div style="font-size: 3rem;">{icon}</div>
            <p style="font-size: 1.1rem; margin-top: 1rem;">{message}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_player_display_name(row: pd.Series) -> str:
    """Get formatted player display name with team.

    Args:
        row: Player row.

    Returns:
        Formatted string.
    """
    name = row.get("player_name", "Unknown")
    team = row.get("team_name", "")
    band = row.get("position_band", "")
    parts = [name]
    if team:
        parts.append(f"({team})")
    if band:
        parts.append(f"[{band}]")
    return " ".join(parts)