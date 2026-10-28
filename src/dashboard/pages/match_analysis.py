"""Match Analysis page for the Premier League Intelligence Dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import load_fixtures_openfootball, load_report_files
from src.dashboard.utils import (
    apply_custom_css,
    format_number,
    show_data_warning,
    show_info_box,
    show_empty_state,
    COLORS,
)


def render_match_analysis_page(source: str = "StatsBomb", season_name: str = "2015/16") -> None:
    """Render the Match Analysis page."""
    apply_custom_css()

    st.title("📊 Match Analysis")
    st.caption(f"Match outcome analysis for {source} {season_name}")

    # Show data limitation warning based on source
    if source == "StatsBomb":
        show_data_warning(
            "Match predictions shown are **historical evaluations** on the 2015/16 season using "
            "chronological rolling-origin validation. They are NOT live predictions for current fixtures. "
            "All models are baselines; the feature-enhanced model does not improve on simple baselines "
            "with single-season data."
        )
    elif source == "OpenFootball":
        show_data_warning(
            "OpenFootball provides match results only (2010/11–2025/26). "
            "No player-level data, xG, or model evaluations available. "
            "2025/26 season has some malformed scores (partial data)."
        )
    elif source == "FPL":
        show_data_warning(
            "FPL provides player gameweek data (2016/17–2025/26). "
            "Match results available via OpenFootball. No model evaluations available from FPL."
        )
    else:
        show_data_warning(
            f"{source} provides match results only. No player-level data or model evaluations available."
        )

    # Load data
    try:
        if source == "StatsBomb":
            from src.dashboard.data_loader import load_fixtures_statsbomb
            fixtures = load_fixtures_statsbomb()
        elif source == "OpenFootball":
            fixtures = load_fixtures_openfootball(season_name)
        elif source == "FPL":
            fixtures = load_fixtures_openfootball(season_name)
        else:
            fixtures = pd.DataFrame()

        reports = load_report_files()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    if fixtures.empty:
        st.warning(f"No match data available for {source} {season_name}.")
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Season Summary",
        "📊 Match Results",
        "🎯 Model Evaluation",
        "📄 Reports",
    ])

    with tab1:
        render_season_summary_tab(fixtures, source, season_name)

    with tab2:
        render_match_results_tab(fixtures, source, season_name)

    with tab3:
        render_model_evaluation_tab(reports)

    with tab4:
        render_reports_tab(reports)


def render_season_summary_tab(fixtures: pd.DataFrame, source: str, season_name: str) -> None:
    """Render season summary tab."""
    st.header(f"📈 {source} {season_name} — Season Summary")

    # Key metrics
    total_matches = len(fixtures)
    total_goals = int(fixtures["home_score_ft"].sum() + fixtures["away_score_ft"].sum())
    home_goals = int(fixtures["home_score_ft"].sum())
    away_goals = int(fixtures["away_score_ft"].sum())
    avg_goals = total_goals / len(fixtures) if len(fixtures) > 0 else 0

    # Match outcomes
    fixtures_copy = fixtures.copy()
    fixtures_copy["outcome"] = fixtures_copy.apply(
        lambda r: "Home Win" if r["home_score_ft"] > r["away_score_ft"]
        else ("Draw" if r["home_score_ft"] == r["away_score_ft"] else "Away Win"),
        axis=1
    )
    outcome_counts = fixtures_copy["outcome"].value_counts()
    home_wins = outcome_counts.get("Home Win", 0)
    draws = outcome_counts.get("Draw", 0)
    away_wins = outcome_counts.get("Away Win", 0)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Matches", format_number(total_matches))
    with col2:
        st.metric("Total Goals", format_number(total_goals))
    with col3:
        st.metric("Avg Goals/Match", f"{total_goals/total_matches:.2f}" if total_matches > 0 else "0.00")
    with col4:
        st.metric("Home Win %", f"{home_wins/total_matches*100:.1f}%" if total_matches > 0 else "0%")
    with col5:
        st.metric("Draw %", f"{draws/total_matches*100:.1f}%" if total_matches > 0 else "0%")

    st.divider()

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Outcome distribution
        outcome_counts = fixtures_copy["outcome"].value_counts()
        fig_pie = px.pie(
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
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Goals distribution
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

    st.divider()

    # Home/Away goals
    col1, col2 = st.columns(2)

    with col1:
        fig_ha = px.box(
            fixtures,
            y=["home_score_ft", "away_score_ft"],
            title="Goals Scored: Home vs Away",
            labels={"value": "Goals", "variable": "Venue"},
            color_discrete_sequence=[COLORS["home"], COLORS["away"]],
        )
        st.plotly_chart(fig_ha, use_container_width=True)

    with col2:
        # Half-time vs Full-time
        fixtures_copy = fixtures.copy()
        fixtures_copy["ht_total"] = fixtures_copy["home_score_ht"] + fixtures_copy["away_score_ht"]
        fixtures_copy["ft_total"] = fixtures_copy["home_score_ft"] + fixtures_copy["away_score_ft"]
        fig_htft = px.scatter(
            fixtures_copy,
            x="ht_total",
            y="ft_total",
            title="Half-Time vs Full-Time Total Goals",
            labels={"ht_total": "HT Total Goals", "ft_total": "FT Total Goals"},
            trendline="ols",
        )
        st.plotly_chart(fig_htft, use_container_width=True)


def render_match_results_tab(fixtures: pd.DataFrame, source: str, season_name: str) -> None:
    """Render match results tab."""
    st.header(f"📊 {season_name} Match Results")

    # Add matchweek filter if available
    if "match_week" in fixtures.columns and fixtures["match_week"].notna().any():
        max_mw = int(fixtures["match_week"].max())
        min_mw = int(fixtures["match_week"].min())
        selected_mw = st.slider(
            "Matchweek",
            min_value=min_mw,
            max_value=max_mw,
            value=(min_mw, max_mw),
        )
        fixtures_filtered = fixtures[
            (fixtures["match_week"] >= selected_mw[0]) & (fixtures["match_week"] <= selected_mw[1])
        ].copy()
    else:
        fixtures_filtered = fixtures.copy()

    # Show match results table
    display_df = fixtures_filtered.copy()
    display_df = display_df.sort_values("match_date" if "match_date" in display_df.columns else "match_week")

    # Select display columns
    cols_to_show = []
    for col in ["match_date", "match_week", "home_team_name", "home_score_ft", "away_score_ft", "away_team_name", "match_week"]:
        if col in display_df.columns:
            cols_to_show.append(col)

    if "match_date" in display_df.columns:
        display_df["match_date"] = pd.to_datetime(display_df["match_date"]).dt.strftime("%Y-%m-%d")

    # Create result column
    display_df["Result"] = display_df.apply(
        lambda r: f"{r['home_score_ft']} - {r['away_score_ft']}", axis=1
    )

    display_cols = ["match_date", "match_week", "home_team_name", "Result", "away_team_name"]
    display_cols = [c for c in display_cols if c in display_df.columns]

    st.dataframe(
        display_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "match_date": "Date",
            "match_week": "MW",
            "home_team_name": "Home",
            "Result": "Result",
            "away_team_name": "Away",
        },
    )

    st.caption(f"Showing {len(display_df)} of {len(fixtures)} matches")


def render_model_evaluation_tab(reports: dict) -> None:
    """Render model evaluation tab."""
    st.header("🎯 Model Evaluation")

    # Only StatsBomb 2015/16 has model evaluations
    st.info("Model evaluations are currently only available for StatsBomb 2015/16 data (Phase 4 evaluation).")

    # Load comparison metrics
    comp_report = None
    for key in ["phase4_stage42_comparison_metrics", "phase4_stage42_report"]:
        if key in reports:
            if reports[key].suffix == ".parquet":
                comp_report = pd.read_parquet(reports[key])
            break

    if comp_report is not None:
        st.subheader("Feature Model vs Baselines (Cutoffs 10-20, 110 matches)")
        st.dataframe(comp_report, use_container_width=True, hide_index=True)

        if "model" in comp_report.columns:
            fig = px.bar(
                comp_report,
                x="model",
                y=["log_loss", "brier_score", "accuracy"],
                title="Feature Model vs Baselines",
                barmode="group",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Detailed metrics available after running Phase 4 Stage 4.2 evaluation.")
        summary_data = {
            "Model": ["Recent Form", "Global Frequencies", "Home Advantage", "Team Strength", "Feature Model"],
            "Log Loss": [1.1033, 1.0893, 1.0893, 1.1499, 1.5156],
            "Brier Score": [0.6577, 0.6585, 0.6585, 0.7142, 0.8142],
            "Accuracy": [0.4182, 0.3727, 0.3727, 0.3273, 0.3273],
        }
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            summary_df,
            x="Model",
            y=["Log Loss", "Brier Score", "Accuracy"],
            title="Model Comparison (Cutoffs 10-20, from documentation)",
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Key finding
    st.warning(
        "**Key Finding**: The feature-enhanced multinomial logistic regression "
        "underperforms all simple baselines on all metrics. "
        "With only one season of data, simple heuristics outperform complex models. "
        "This is expected and honestly reported."
    )


def render_reports_tab(reports: dict) -> None:
    """Render reports tab."""
    st.header("📄 Available Reports")

    if not reports:
        st.info("No reports found. Run the evaluation pipeline to generate reports.")
        return

    match_reports = {k: v for k, v in reports.items() if "phase4" in k or "match" in k}

    if not match_reports:
        st.info("No match-specific reports found. Showing all reports.")
        match_reports = reports

    selected = st.selectbox("Select Report", sorted(match_reports.keys()))

    if selected:
        path = match_reports[selected]
        if path.suffix == ".md":
            st.markdown(path.read_text())
        elif path.suffix == ".parquet":
            df = pd.read_parquet(path)
            st.dataframe(df, use_container_width=True)
        else:
            st.info(f"Report format {path.suffix} not directly viewable.")


if __name__ == "__main__":
    render_match_analysis_page()