"""League Simulator page for the Premier League Intelligence Dashboard."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import load_fixtures
from src.models.league_simulator import (
    SimulatorConfig,
    run_league_simulation,
    make_global_frequency_provider,
    make_recent_form_provider,
    make_scoreline_poisson_provider,
    validate_fixtures,
)
from src.dashboard.utils import (
    apply_custom_css,
    format_number,
    show_data_warning,
    show_info_box,
    show_empty_state,
    COLORS,
)


def render_league_simulator_page() -> None:
    """Render the League Simulator page."""
    apply_custom_css()

    st.title("🏆 League Simulator")
    st.caption("Monte Carlo simulation of the 2015/16 Premier League season")

    show_data_warning(
        "Simulations use historical 2015/16 data and probability providers evaluated on that season. "
        "Results show the range of possible outcomes given the model assumptions — they are NOT "
        "forecasts for the current season. The independent Poisson assumption for scoreline mode "
        "is a known limitation (real matches show goal dependence)."
    )

    # Load fixtures
    try:
        fixtures = load_fixtures()
    except Exception as e:
        st.error(f"Failed to load fixtures: {e}")
        return

    # Validate fixtures
    try:
        validate_fixtures(fixtures)
    except Exception as e:
        st.error(f"Fixture validation failed: {e}")
        return

    # Check for played/unplayed matches
    played = fixtures[fixtures["home_score"].notna() & fixtures["away_score"].notna()]
    unplayed = fixtures[fixtures["home_score"].isna() | fixtures["away_score"].isna()]

    st.info(f"Season: 380 matches total | Played: {len(played)} | Remaining: {len(unplayed)}")

    # Simulation configuration
    with st.expander("⚙️ Simulation Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Provider**")
            provider_type = st.selectbox(
                "Probability Provider",
                options=["global_frequency", "recent_form", "scoreline_poisson"],
                format_func=lambda x: {
                    "global_frequency": "Global Frequencies (max entropy baseline)",
                    "recent_form": "Recent Form (team-specific, best predictor)",
                    "scoreline_poisson": "Scoreline Poisson (goal-level, slower)",
                }[x],
            )

            if provider_type == "recent_form":
                form_window = st.slider("Form Window (matches)", 3, 10, 5)
            elif provider_type == "scoreline_poisson":
                home_adv = st.number_input("Home Advantage (goals)", 0.0, 1.0, 0.15, 0.05)
                min_hist = st.number_input("Min History for Team Rates", 1, 20, 5)

        with col2:
            st.markdown("**Simulation**")
            n_simulations = st.selectbox(
                "Number of Simulations",
                options=[100, 500, 1000, 2000, 5000, 10000],
                index=2,  # 1000 default
                format_func=lambda x: f"{x:,}",
            )
            random_seed = st.number_input("Random Seed", value=42, step=1)
            sim_mode = st.radio(
                "Simulation Mode",
                options=["outcome", "scoreline"],
                format_func=lambda x: "Outcome (W/D/L)" if x == "outcome" else "Scoreline (exact goals)",
                disabled=(provider_type != "scoreline_poisson"),
            )

        with col3:
            st.markdown("**League Rules**")
            points_win = st.number_input("Points for Win", 2, 5, 3)
            points_draw = st.number_input("Points for Draw", 0, 2, 1)
            n_relegation = st.number_input("Relegation Places", 2, 4, 3)
            n_top = st.number_input("Top Places (UCL)", 2, 6, 4)
            tiebreaker = st.selectbox(
                "Tiebreaker",
                options=["points_only", "goal_diff_then_goals"],
                format_func=lambda x: "Points Only" if x == "points_only" else "GD then GS",
            )

    # Run simulation button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        run_simulation(
            fixtures=fixtures,
            provider_type=provider_type,
            n_simulations=n_simulations,
            random_seed=random_seed,
            sim_mode=sim_mode if provider_type == "scoreline_poisson" else "outcome",
            points_win=points_win,
            points_draw=points_draw,
            n_relegation=n_relegation,
            n_top=n_top,
            tiebreaker=tiebreaker,
            form_window=form_window if provider_type == "recent_form" else 5,
            home_adv=home_adv if provider_type == "scoreline_poisson" else 0.15,
            min_hist=min_hist if provider_type == "scoreline_poisson" else 5,
        )


def run_simulation(
    fixtures: pd.DataFrame,
    provider_type: str,
    n_simulations: int,
    random_seed: int,
    sim_mode: str,
    points_win: int,
    points_draw: int,
    n_relegation: int,
    n_top: int,
    tiebreaker: str,
    form_window: int = 5,
    home_adv: float = 0.15,
    min_hist: int = 5,
) -> None:
    """Run the league simulation and display results.

    Args:
        fixtures: Fixtures DataFrame.
        provider_type: Type of probability provider.
        n_simulations: Number of Monte Carlo simulations.
        random_seed: Random seed for reproducibility.
        sim_mode: Simulation mode (outcome or scoreline).
        points_win: Points for a win.
        points_draw: Points for a draw.
        n_relegation: Number of relegation places.
        n_top: Number of top places.
        tiebreaker: Tiebreaker method.
        form_window: Window for recent form provider.
        home_adv: Home advantage for Poisson provider.
        min_hist: Minimum history for team-specific rates.
    """
    config = SimulatorConfig(
        n_simulations=n_simulations,
        random_seed=random_seed,
        points_win=points_win,
        points_draw=points_draw,
        n_relegation_places=n_relegation,
        n_top_places=n_top,
        tiebreaker=tiebreaker,
        simulation_mode=sim_mode,
    )

    # Create probability provider
    if provider_type == "global_frequency":
        provider = make_global_frequency_provider(fixtures)
    elif provider_type == "recent_form":
        provider = make_recent_form_provider(fixtures, window=form_window)
    elif provider_type == "scoreline_poisson":
        provider = make_scoreline_poisson_provider(
            fixtures,
            smoothing_alpha=1.0,
            home_advantage=home_adv,
            min_history=min_hist,
        )
    else:
        st.error(f"Unknown provider type: {provider_type}")
        return

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    def progress_callback(completed: int, total: int):
        progress_bar.progress(completed / total)
        status_text.text(f"Simulation {completed}/{total} ({completed/total*100:.1f}%)")

    # Run simulation
    start_time = time.time()

    try:
        # We need to wrap the provider to add progress tracking
        # For now, run without callback (the simulator doesn't support it yet)
        status_text.text("Running simulation...")

        results = run_league_simulation(
            fixtures=fixtures,
            probability_provider=provider,
            config=config,
        )

        elapsed = time.time() - start_time
        progress_bar.progress(1.0)
        status_text.text(f"✅ Completed in {elapsed:.1f}s")

        display_results(results, config)

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Simulation failed: {e}")
        import traceback
        st.code(traceback.format_exc())


def display_results(results: dict, config: SimulatorConfig) -> None:
    """Display simulation results.

    Args:
        results: Results dictionary from run_league_simulation.
        config: SimulatorConfig used.
    """
    st.divider()
    st.header("📊 Simulation Results")

    stats = results["statistics"]
    teams = results["teams"]
    all_standings = results["standings"]

    # Load team names for display
    fixtures = load_fixtures()
    team_id_to_name = {}
    for _, row in fixtures.iterrows():
        team_id_to_name[row["home_team_id"]] = row["home_team_name"]
        team_id_to_name[row["away_team_id"]] = row["away_team_name"]

    # Title probabilities
    st.subheader("🏆 Title Probabilities")
    title_probs = stats["title_probabilities"]
    title_df = pd.DataFrame({
        "Team": [team_id_to_name.get(tid, f"Team {tid}") for tid in title_probs.index],
        "Probability (%)": (title_probs * 100).round(1),
    })
    title_df = title_df[title_df["Probability (%)"] > 0].head(10)

    fig_title = px.bar(
        title_df,
        x="Probability (%)",
        y="Team",
        orientation="h",
        title="Title Win Probability",
        color="Probability (%)",
        color_continuous_scale="Blues",
    )
    fig_title.update_layout(yaxis=dict(autorange="reversed"), height=400)
    st.plotly_chart(fig_title, use_container_width=True)

    # Top 4 probabilities
    st.subheader("🥇 Top 4 Probabilities")
    top_probs = stats["top_n_probabilities"]
    top_df = pd.DataFrame({
        "Team": [team_id_to_name.get(tid, f"Team {tid}") for tid in top_probs.index],
        "Probability (%)": (top_probs * 100).round(1),
    })
    top_df = top_df.sort_values("Probability (%)", ascending=False)

    fig_top = px.bar(
        top_df,
        x="Probability (%)",
        y="Team",
        orientation="h",
        title="Top 4 Finish Probability",
        color="Probability (%)",
        color_continuous_scale="Greens",
    )
    fig_top.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig_top, use_container_width=True)

    # Relegation probabilities
    st.subheader("📉 Relegation Probabilities")
    releg_probs = stats["relegation_probabilities"]
    releg_df = pd.DataFrame({
        "Team": [team_id_to_name.get(tid, f"Team {tid}") for tid in releg_probs.index],
        "Probability (%)": (releg_probs * 100).round(1),
    })
    releg_df = releg_df.sort_values("Probability (%)", ascending=False)

    fig_releg = px.bar(
        releg_df,
        x="Probability (%)",
        y="Team",
        orientation="h",
        title="Relegation Probability",
        color="Probability (%)",
        color_continuous_scale="Reds",
    )
    fig_releg.update_layout(yaxis=dict(autorange="reversed"), height=500)
    st.plotly_chart(fig_releg, use_container_width=True)

    # Expected points table
    st.subheader("📋 Expected Final Table")
    exp_points = stats["expected_points"]
    exp_position = stats["expected_position"]

    table_df = pd.DataFrame({
        "Team": [team_id_to_name.get(tid, f"Team {tid}") for tid in exp_points.index],
        "Expected Points": exp_points.round(1),
        "Expected Position": exp_position.round(1),
        "Title %": (stats["title_probabilities"].reindex(exp_points.index).fillna(0) * 100).round(1),
        "Top 4 %": (stats["top_n_probabilities"].reindex(exp_points.index).fillna(0) * 100).round(1),
        "Releg %": (stats["relegation_probabilities"].reindex(exp_points.index).fillna(0) * 100).round(1),
    })
    table_df = table_df.sort_values("Expected Points", ascending=False).reset_index(drop=True)
    table_df.index += 1
    table_df.index.name = "Exp. Pos"

    st.dataframe(table_df, use_container_width=True)

    # Position distributions for selected teams
    st.subheader("📈 Position Distributions")
    pos_dist = stats["position_distribution"]

    # Select teams to show
    top_teams = table_df.head(6)["Team"].tolist()
    bottom_teams = table_df.tail(6)["Team"].tolist()
    selected_teams = st.multiselect(
        "Select teams to show position distribution",
        options=table_df["Team"].tolist(),
        default=top_teams + bottom_teams[:3],
    )

    if selected_teams:
        team_ids = [tid for tid, name in team_id_to_name.items() if name in selected_teams]
        dist_data = pos_dist.loc[pos_dist.index.isin(team_ids)].T
        dist_data.columns = [team_id_to_name.get(c, f"Team {c}") for c in dist_data.columns]

        fig_dist = px.line(
            dist_data,
            title="Position Distribution by Team",
            labels={"index": "Final Position", "value": "Probability", "variable": "Team"},
        )
        fig_dist.update_traces(mode="lines+markers")
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Simulation details
    with st.expander("Simulation Details"):
        st.json({
            "n_simulations": config.n_simulations,
            "random_seed": config.random_seed,
            "simulation_mode": config.simulation_mode,
            "points_win": config.points_win,
            "points_draw": config.points_draw,
            "n_relegation_places": config.n_relegation_places,
            "n_top_places": config.n_top_places,
            "tiebreaker": config.tiebreaker,
        })


if __name__ == "__main__":
    render_league_simulator_page()