"""Overview page for the Premier League Intelligence Dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import (
    load_fixtures,
    load_player_season,
    get_season_info,
    load_report_files,
)
from src.dashboard.utils import (
    apply_custom_css,
    format_number,
    show_data_warning,
    show_info_box,
    COLORS,
)


def render_overview_page() -> None:
    """Render the Overview page."""
    apply_custom_css()

    st.title("⚽ Premier League Intelligence")
    st.caption("Historical analysis of the 2015/16 Premier League season")

    # Show data limitation warning
    show_data_warning(
        "This dashboard presents historical analysis of the 2015/16 Premier League season "
        "using StatsBomb Open Data. All model outputs, simulations, and scouting indicators "
        "are retrospective and based on a single season of data. They should not be used "
        "for current-season forecasting or financial decisions."
    )

    # Load data
    try:
        season_info = get_season_info()
        fixtures = load_fixtures()
        player_season = load_player_season()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.info("Please run the data ingestion pipeline first. See README for instructions.")
        return

    # Season header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Season", season_info["season_name"])
    with col2:
        st.metric("Matches", format_number(season_info["n_matches"]))
    with col3:
        st.metric("Teams", format_number(season_info["n_teams"]))
    with col4:
        st.metric("Players", format_number(season_info["n_players"]))

    st.divider()

    # Key season statistics
    st.header("📊 Key Season Statistics")

    col1, col2 = st.columns(2)

    with col1:
        # Goals distribution
        total_goals = int(fixtures["home_score"].sum() + fixtures["away_score"].sum())
        home_goals = int(fixtures["home_score"].sum())
        away_goals = int(fixtures["away_score"].sum())
        avg_goals = total_goals / len(fixtures)

        st.subheader("Goals")
        g1, g2, g3 = st.columns(3)
        g1.metric("Total Goals", format_number(total_goals))
        g2.metric("Home Goals", format_number(home_goals))
        g3.metric("Away Goals", format_number(away_goals))
        st.caption(f"Average: {avg_goals:.2f} goals per match")

        # Goals per match histogram
        fixtures_copy = fixtures.copy()
        fixtures_copy["total_goals"] = fixtures_copy["home_score"] + fixtures_copy["away_score"]
        fig_goals = px.histogram(
            fixtures_copy,
            x="total_goals",
            nbins=10,
            title="Goals per Match Distribution",
            labels={"total_goals": "Total Goals", "count": "Matches"},
            color_discrete_sequence=[COLORS["primary"]],
        )
        fig_goals.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_goals, use_container_width=True)

    with col2:
        # Match outcomes
        fixtures_copy = fixtures.copy()
        fixtures_copy["outcome"] = fixtures_copy.apply(
            lambda r: "Home Win" if r["home_score"] > r["away_score"]
            else ("Draw" if r["home_score"] == r["away_score"] else "Away Win"),
            axis=1
        )
        outcome_counts = fixtures_copy["outcome"].value_counts()

        st.subheader("Match Outcomes")
        o1, o2, o3 = st.columns(3)
        o1.metric("Home Wins", format_number(outcome_counts.get("Home Win", 0)))
        o2.metric("Draws", format_number(outcome_counts.get("Draw", 0)))
        o3.metric("Away Wins", format_number(outcome_counts.get("Away Win", 0)))

        fig_outcomes = px.pie(
            values=outcome_counts.values,
            names=outcome_counts.index,
            title="Match Outcome Distribution",
            color=outcome_counts.index,
            color_discrete_map={
                "Home Win": COLORS["home"],
                "Draw": COLORS["draw"],
                "Away Win": COLORS["away"],
            },
        )
        fig_outcomes.update_layout(height=300)
        st.plotly_chart(fig_outcomes, use_container_width=True)

    st.divider()

    # Top scorers
    st.header("🥅 Top Scorers (2015/16)")
    top_scorers = player_season.nlargest(10, "goals")[
        ["player_name", "team_name", "position", "goals", "assists", "xg", "minutes"]
    ].reset_index(drop=True)
    top_scorers.index += 1
    st.dataframe(
        top_scorers.rename(columns={
            "player_name": "Player",
            "team_name": "Team",
            "position": "Position",
            "goals": "Goals",
            "assists": "Assists",
            "xg": "xG",
            "minutes": "Minutes",
        }),
        use_container_width=True,
        hide_index=False,
    )

    st.divider()

    # Navigation guide
    st.header("🧭 Dashboard Navigation")
    st.markdown("""
    Use the sidebar to navigate between dashboard sections:

    | Section | Description |
    |---------|-------------|
    | **Overview** | This page — dataset coverage and season summary |
    | **Player Explorer** | Search and filter players; view per-90 metrics and profiles |
    | **Scouting & Comparisons** | Compare players side-by-side; view position percentiles; configure scouting priority weights |
    | **Match Analysis** | View evaluated match outcome baselines and model performance |
    | **League Simulator** | Run Monte Carlo simulations of the league table |
    """)

    st.divider()

    # Available reports
    st.header("📄 Available Reports")
    reports = load_report_files()
    if reports:
        report_names = sorted(reports.keys())
        selected_report = st.selectbox("Select a report to view", report_names)
        if selected_report:
            report_path = reports[selected_report]
            if report_path.suffix == ".md":
                st.markdown(report_path.read_text())
            else:
                st.info("Parquet report selected — use a data viewer to inspect.")
    else:
        st.info("No reports found in the reports directory.")


if __name__ == "__main__":
    render_overview_page()