"""Overview page for the Premier League Intelligence Dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import (
    get_season_info,
    get_season_coverage_summary,
    load_report_files,
)
from src.dashboard.utils import (
    apply_custom_css,
    format_number,
    show_data_warning,
    show_info_box,
    COLORS,
)


def render_overview_page(source: str = "StatsBomb", season_name: str = "2015/16") -> None:
    """Render the Overview page with multi-season, multi-source coverage."""
    apply_custom_css()

    st.title("⚽ Premier League Intelligence")
    st.caption("Multi-Season Premier League Analysis")

    # Show data limitation warning
    show_data_warning(
        "This dashboard presents historical analysis of Premier League seasons using multiple data sources. "
        "All model outputs, simulations, and scouting indicators are retrospective. "
        "They should not be used for current-season forecasting or financial decisions."
    )

    # Load coverage summary
    try:
        coverage_df = get_season_coverage_summary()
    except Exception as e:
        st.error(f"Failed to load coverage data: {e}")
        st.info("Please run the multi-season ingestion pipeline first. See README for instructions.")
        return

    # Season header with coverage metrics
    st.header("📅 Season Coverage Overview")

    # Filter coverage for selected source
    source_df = coverage_df[coverage_df["Source"] == source].copy()
    if source_df.empty:
        st.warning(f"No data available for {source}")
        return

    # Show season coverage table
    st.subheader(f"📋 {source} Coverage by Season")

    display_cols = ["Season", "Matches", "Teams", "Players", "Player Match Rows", "Player Season Rows", "xG", "xA", "Note"]
    available_cols = [c for c in display_cols if c in source_df.columns]

    # Format for display
    display_df = source_df[available_cols].copy()
    for col in ["xG", "xA"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].map({True: "✅", False: "❌"})

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # Aggregate metrics for selected source
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_matches = source_df["Matches"].sum()
        st.metric("Total Matches", format_number(total_matches))
    with col2:
        avg_teams = int(source_df["Teams"].mean()) if len(source_df) > 0 else 0
        st.metric("Avg Teams/Season", format_number(avg_teams))
    with col3:
        total_players = source_df["Players"].sum()
        st.metric("Total Players", format_number(total_players))
    with col4:
        xg_seasons = source_df["xG"].sum() if "xG" in source_df.columns else 0
        st.metric("Seasons with xG", format_number(int(xg_seasons)))

    st.divider()

    # Season-level trends
    st.header("📈 Season-Level Trends")

    col1, col2 = st.columns(2)

    with col1:
        # Matches per season
        fig_matches = px.bar(
            source_df.sort_values("Season"),
            x="Season",
            y="Matches",
            title="Matches per Season",
            color="Matches",
            color_continuous_scale="Blues",
        )
        fig_matches.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_matches, use_container_width=True)

    with col2:
        # xG/xA availability
        if "xG" in source_df.columns and "xA" in source_df.columns:
            trend_df = source_df[["Season", "xG", "xA"]].copy()
            trend_df["xG"] = trend_df["xG"].astype(int)
            trend_df["xA"] = trend_df["xA"].astype(int)
            trend_df = trend_df.melt(id_vars="Season", value_vars=["xG", "xA"], var_name="Metric", value_name="Available")

            fig_xg = px.bar(
                trend_df,
                x="Season",
                y="Available",
                color="Metric",
                title="xG/xA Availability by Season",
                barmode="group",
                color_discrete_map={"xG": "#1a5c9e", "xA": "#e87722"},
            )
            fig_xg.update_layout(height=350)
            st.plotly_chart(fig_xg, use_container_width=True)

    st.divider()

    # Selected season detail
    st.header(f"📊 {source} — {season_name} Detail")

    try:
        from src.dashboard.data_loader import get_season_info
        season_info = get_season_info(source, season_name)
    except Exception as e:
        st.error(f"Failed to load season info: {e}")
        season_info = {"n_matches": 0, "n_teams": 0, "n_players": 0}

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Season", season_name)
    with col2:
        st.metric("Matches", format_number(season_info.get("n_matches", 0)))
    with col3:
        st.metric("Teams", format_number(season_info.get("n_teams", 0)))
    with col4:
        st.metric("Players", format_number(season_info.get("n_players", 0)))

    # Load fixtures for selected season/source
    try:
        if source == "StatsBomb":
            from src.dashboard.data_loader import load_fixtures_statsbomb
            fixtures = load_fixtures_statsbomb()
            if season_name != "2015/16":
                st.info(f"StatsBomb only has 2015/16 data. Showing 2015/16 fixtures.")
                fixtures = load_fixtures_statsbomb()
        elif source == "OpenFootball":
            from src.dashboard.data_loader import load_fixtures_openfootball
            fixtures = load_fixtures_openfootball(season_name)
        elif source == "FPL":
            from src.dashboard.data_loader import load_fixtures_openfootball
            fixtures = load_fixtures_openfootball(season_name)
        else:
            fixtures = pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to load fixtures: {e}")
        fixtures = pd.DataFrame()

    if len(fixtures) > 0:
        col1, col2 = st.columns(2)

        with col1:
            # Goals distribution
            total_goals = int(fixtures["home_score_ft"].sum() + fixtures["away_score_ft"].sum())
            home_goals = int(fixtures["home_score_ft"].sum())
            away_goals = int(fixtures["away_score_ft"].sum())
            avg_goals = total_goals / len(fixtures) if len(fixtures) > 0 else 0

            st.subheader("Goals")
            g1, g2, g3 = st.columns(3)
            g1.metric("Total Goals", format_number(total_goals))
            g2.metric("Home Goals", format_number(home_goals))
            g3.metric("Away Goals", format_number(away_goals))
            st.caption(f"Average: {avg_goals:.2f} goals per match")

            # Goals per match histogram
            fixtures_copy = fixtures.copy()
            fixtures_copy["total_goals"] = fixtures_copy["home_score_ft"] + fixtures_copy["away_score_ft"]
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
                lambda r: "Home Win" if r["home_score_ft"] > r["away_score_ft"]
                else ("Draw" if r["home_score_ft"] == r["away_score_ft"] else "Away Win"),
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

    # Data source comparison
    st.header("🔄 Data Source Comparison")
    show_info_box(
        "Source Boundaries",
        "**StatsBomb**: Event-level data (2015/16 only) — goals, xG, passes, pressures, etc.\n\n"
        "**OpenFootball**: Match results only (2010/11–2025/26) — scores, no player stats.\n\n"
        "**FPL**: Player gameweek stats (2016/17–2025/26) — minutes, goals, assists, xG/xA (2022/23+), fantasy points.\n\n"
        "These sources are **not merged** at the player level. Player IDs differ across sources. "
        "Do not compare StatsBomb xG with FPL xG directly — they use different methodologies."
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
    from src.dashboard.data_loader import load_report_files
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