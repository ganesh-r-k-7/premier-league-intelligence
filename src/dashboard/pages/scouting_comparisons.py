"""Scouting & Comparisons page for the Premier League Intelligence Dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import load_player_profiles, get_comparison_metrics
from src.features.player_comparison import PlayerComparison, ComparisonConfig
from src.features.scouting_priority import calculate_scouting_priority, PriorityConfig
from src.features.player_scouting import compute_similar_players, ScoutingConfig
from src.dashboard.utils import (
    apply_custom_css,
    format_number,
    create_radar_chart,
    create_bar_comparison,
    show_data_warning,
    show_empty_state,
    show_info_box,
    get_player_display_name,
    COLORS,
    POSITION_BAND_COLORS,
)


def render_scouting_page() -> None:
    """Render the Scouting & Comparisons page."""
    apply_custom_css()

    st.title("🎯 Scouting & Comparisons")
    st.caption("Compare players, view position percentiles, and configure scouting priorities")

    show_data_warning(
        "Scouting indicators are performance-based heuristics computed on 2015/16 data only. "
        "They do not incorporate market values, age, contract status, injury history, or "
        "multi-season consistency. These are not financial valuations or transfer recommendations."
    )

    # Load data
    try:
        profiles = load_player_profiles()
    except Exception as e:
        st.error(f"Failed to load player data: {e}")
        return

    # Tabs for different scouting functions
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Player Comparison",
        "📈 Position Percentiles",
        "🎯 Scouting Priority",
        "🔄 Similar Players",
    ])

    with tab1:
        render_player_comparison_tab(profiles)

    with tab2:
        render_position_percentiles_tab(profiles)

    with tab3:
        render_scouting_priority_tab(profiles)

    with tab4:
        render_similar_players_tab(profiles)


def render_player_comparison_tab(profiles: pd.DataFrame) -> None:
    """Render the player comparison tab."""
    st.header("Side-by-Side Player Comparison")

    # Player selection
    col1, col2 = st.columns(2)
    player_options = profiles.apply(get_player_display_name, axis=1).tolist()
    player_ids = profiles["player_id"].tolist()

    with col1:
        p1_idx = st.selectbox(
            "Player 1",
            options=range(len(player_options)),
            format_func=lambda i: player_options[i],
            key="comp_p1",
        )
    with col2:
        p2_idx = st.selectbox(
            "Player 2",
            options=range(len(player_options)),
            format_func=lambda i: player_options[i],
            index=1 if len(player_options) > 1 else 0,
            key="comp_p2",
        )

    if p1_idx == p2_idx:
        st.warning("Please select two different players.")
        return

    p1_id = player_ids[p1_idx]
    p2_id = player_ids[p2_idx]

    # Metric selection
    available_metrics = get_comparison_metrics()
    metric_cols = [m[0] for m in available_metrics]
    metric_labels = {m[0]: m[1] for m in available_metrics}

    default_metrics = [
        "goals_per_90", "assists_per_90", "xg_per_90", "xa_per_90",
        "shots_per_90", "goals_per_shot", "goals_over_xg", "minutes",
    ]
    selected_metrics = st.multiselect(
        "Metrics to compare",
        options=metric_cols,
        default=[m for m in default_metrics if m in metric_cols],
        format_func=lambda x: metric_labels.get(x, x),
    )

    if not selected_metrics:
        st.info("Select at least one metric to compare.")
        return

    # Configuration
    with st.expander("Comparison Settings"):
        col1, col2 = st.columns(2)
        with col1:
            min_minutes = st.number_input("Min minutes for percentile", value=450, step=90)
            percentile_scope = st.radio(
                "Percentile scope",
                options=["position_band", "global"],
                format_func=lambda x: "Position Band" if x == "position_band" else "Global",
            )
        with col2:
            include_percentiles = st.checkbox("Include percentiles", value=True)
            handle_missing = st.selectbox(
                "Handle missing values",
                options=["flag", "exclude", "zero"],
                format_func=lambda x: x.capitalize(),
            )

    # Run comparison
    config = ComparisonConfig(
        min_minutes=min_minutes,
        percentile_scope=percentile_scope,
        include_percentiles=include_percentiles,
        handle_missing=handle_missing,
    )

    comparison = PlayerComparison(profiles, config)
    result = comparison.compare_players([p1_id, p2_id], selected_metrics)

    p1_name = profiles.loc[profiles["player_id"] == p1_id, "player_name"].values[0]
    p2_name = profiles.loc[profiles["player_id"] == p2_id, "player_name"].values[0]

    st.subheader(f"{p1_name} vs {p2_name}")

    # Display comparison table
    comp_df = result.comparison_df
    display_cols = ["player_name", "team_name", "position_band"]
    for m in selected_metrics:
        display_cols.append(f"{m}_raw")
        if include_percentiles:
            display_cols.append(f"{m}_percentile")
        if len([p1_id, p2_id]) == 2:
            display_cols.append(f"{m}_diff")
            display_cols.append(f"{m}_pct_diff")

    display_df = comp_df[display_cols].copy()

    # Rename columns for display
    rename_map = {"player_name": "Player", "team_name": "Team", "position_band": "Band"}
    for m in selected_metrics:
        rename_map[f"{m}_raw"] = metric_labels.get(m, m)
        if include_percentiles:
            rename_map[f"{m}_percentile"] = f"{metric_labels.get(m, m)} %ile"
        if len([p1_id, p2_id]) == 2:
            rename_map[f"{m}_diff"] = f"Diff ({p1_name} - {p2_name})"
            rename_map[f"{m}_pct_diff"] = f"% Diff"

    display_df = display_df.rename(columns=rename_map)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Radar chart
    st.subheader("Radar Chart Comparison")
    radar_metrics = [m for m in selected_metrics if m.endswith("_per_90") or m in ["goals_per_shot", "goals_over_xg"]]
    if radar_metrics:
        radar_df = profiles[profiles["player_id"].isin([p1_id, p2_id])].copy()
        fig = create_radar_chart(radar_df, metrics=radar_metrics)
        st.plotly_chart(fig, use_container_width=True)

    # Summary
    if not result.summary_df.empty:
        st.subheader("Summary")
        summary = result.summary_df.copy()
        summary = summary.rename(columns={
            "metric": "Metric",
            f"{p1_id}_value": p1_name,
            f"{p2_id}_value": p2_name,
            "difference": "Difference",
            "pct_diff": "% Diff",
            "higher": "Higher",
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)


def render_position_percentiles_tab(profiles: pd.DataFrame) -> None:
    """Render the position percentiles tab."""
    st.header("Position-Band Percentile Rankings")

    # Position band selector
    bands = ["GK", "DEF", "MID", "WING", "FWD"]
    selected_band = st.selectbox("Position Band", bands)

    band_profiles = profiles[
        (profiles["position_band"] == selected_band) &
        (profiles["minutes"] >= 270)
    ].copy()

    if band_profiles.empty:
        show_empty_state(f"No players found in {selected_band} with sufficient minutes.")
        return

    # Metric selector
    available_metrics = get_comparison_metrics()
    rate_metrics = [m[0] for m in available_metrics if m[0].endswith("_per_90") or m[0] in ["goals_per_shot", "goals_over_xg"]]
    metric_labels = {m[0]: m[1] for m in available_metrics}

    selected_metric = st.selectbox(
        "Metric",
        options=rate_metrics,
        format_func=lambda x: metric_labels.get(x, x),
    )

    # Min minutes filter
    min_min = st.slider("Minimum minutes", 0, 3000, 450, 90)
    band_profiles = band_profiles[band_profiles["minutes"] >= min_min]

    if band_profiles.empty:
        show_empty_state("No players meet the minimum minutes threshold.")
        return

    # Compute percentiles
    band_profiles = band_profiles.copy()
    band_profiles["percentile"] = band_profiles[selected_metric].rank(pct=True, method="min") * 100
    band_profiles = band_profiles.sort_values("percentile", ascending=False)

    # Top 20
    st.subheader(f"Top 20 {selected_band} by {metric_labels.get(selected_metric, selected_metric)}")
    top20 = band_profiles.head(20)[
        ["player_name", "team_name", "minutes", selected_metric, "percentile"]
    ].copy()
    top20[selected_metric] = top20[selected_metric].apply(lambda x: f"{x:.2f}")
    top20["percentile"] = top20["percentile"].apply(lambda x: f"{x:.0f}th")
    top20 = top20.rename(columns={
        "player_name": "Player",
        "team_name": "Team",
        "minutes": "Min",
        selected_metric: metric_labels.get(selected_metric, selected_metric),
        "percentile": "Percentile",
    })
    st.dataframe(top20.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Bar chart
    st.subheader("Visualization")
    fig = create_bar_comparison(
        band_profiles.head(20),
        selected_metric,
        color_col="position_band",
        title=f"{metric_labels.get(selected_metric, selected_metric)} — {selected_band} (Top 20)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution
    st.subheader("Distribution")
    fig_dist = px.histogram(
        band_profiles,
        x=selected_metric,
        nbins=20,
        title=f"Distribution of {metric_labels.get(selected_metric, selected_metric)} for {selected_band}",
        color_discrete_sequence=[POSITION_BAND_COLORS.get(selected_band, COLORS["primary"])],
    )
    fig_dist.add_vline(
        x=band_profiles[selected_metric].median(),
        line_dash="dash",
        line_color="red",
        annotation_text="Median",
    )
    st.plotly_chart(fig_dist, use_container_width=True)


def render_scouting_priority_tab(profiles: pd.DataFrame) -> None:
    """Render the scouting priority tab."""
    st.header("Scouting Priority Scores")
    show_info_box(
        "Scouting Priority",
        "This is a **performance-based ranking** combining production vs peers, xG vs actual, "
        "playing time reliability, and optional similarity to a target player. "
        "**It is not a market valuation or transfer recommendation.** "
        "Weights are configurable below."
    )

    # Configuration
    with st.expander("Priority Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Component Weights**")
            w_prod = st.slider("Production vs Peers", 0.0, 1.0, 0.40, 0.05)
            w_xg = st.slider("xG vs Actual", 0.0, 1.0, 0.25, 0.05)
        with col2:
            w_pt = st.slider("Playing Time Reliability", 0.0, 1.0, 0.20, 0.05)
            w_sim = st.slider("Similarity to Target", 0.0, 1.0, 0.15, 0.05)
        with col3:
            min_minutes = st.number_input("Min Minutes", 0, 3000, 450, 90)
            # Target player for similarity
            player_options = profiles.apply(get_player_display_name, axis=1).tolist()
            player_ids = profiles["player_id"].tolist()
            target_idx = st.selectbox(
                "Target Player (for similarity)",
                options=[None] + list(range(len(player_options))),
                format_func=lambda i: "None" if i is None else player_options[i],
            )
            target_id = None if target_idx is None else player_ids[target_idx]

    # Validate weights
    weights = {
        "production_vs_peers": w_prod,
        "xg_vs_actual": w_xg,
        "playing_time_reliability": w_pt,
        "similarity_to_target": w_sim,
    }
    total = sum(weights.values())
    if total == 0:
        st.error("At least one weight must be positive.")
        return

    # Normalize weights for display
    norm_weights = {k: v/total for k, v in weights.items()}
    st.caption(f"Normalized weights: " + ", ".join(f"{k}={v:.0%}" for k, v in norm_weights.items()))

    # Calculate priority
    config = PriorityConfig(
        weights=weights,
        min_minutes=min_minutes,
        target_player_id=target_id,
    )

    with st.spinner("Calculating scouting priorities..."):
        priority_df = calculate_scouting_priority(profiles, config)

    if priority_df.empty:
        show_empty_state("No players meet the minimum minutes threshold.")
        return

    # Display results
    st.subheader(f"Top 30 Scouting Priorities (min {min_minutes} min)")

    display_cols = [
        "priority_rank", "player_name", "team_name", "position_band",
        "minutes", "priority_score", "coverage",
        "production_vs_peers", "xg_vs_actual", "playing_time_reliability",
    ]
    if target_id:
        display_cols.append("similarity_to_target")

    display_df = priority_df.head(30)[display_cols].copy()
    display_df = display_df.rename(columns={
        "priority_rank": "Rank",
        "player_name": "Player",
        "team_name": "Team",
        "position_band": "Band",
        "minutes": "Min",
        "priority_score": "Priority",
        "coverage": "Coverage %",
        "production_vs_peers": "Prod vs Peers",
        "xg_vs_actual": "xG vs Actual",
        "playing_time_reliability": "PT Reliability",
        "similarity_to_target": "Similarity",
    })

    # Format
    for col in ["Priority", "Coverage %", "Prod vs Peers", "xG vs Actual", "PT Reliability", "Similarity"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Visualization
    st.subheader("Priority Score Distribution")
    fig = px.histogram(
        priority_df,
        x="priority_score",
        nbins=20,
        title="Distribution of Scouting Priority Scores",
        color_discrete_sequence=[COLORS["primary"]],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Component breakdown for top player
    top_player = priority_df.iloc[0]
    st.subheader(f"Component Breakdown: {top_player['player_name']} (Rank 1)")
    components = ["production_vs_peers", "xg_vs_actual", "playing_time_reliability"]
    if "similarity_to_target" in priority_df.columns:
        components.append("similarity_to_target")

    comp_data = []
    for comp in components:
        if comp in priority_df.columns:
            comp_data.append({
                "Component": comp.replace("_", " ").title(),
                "Value": top_player[comp],
                "Weight": norm_weights.get(comp, 0),
            })
    comp_df = pd.DataFrame(comp_data)

    fig_comp = px.bar(
        comp_df,
        x="Component",
        y="Value",
        title=f"Priority Components for {top_player['player_name']}",
        color="Component",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig_comp, use_container_width=True)


def render_similar_players_tab(profiles: pd.DataFrame) -> None:
    """Render the similar players tab."""
    st.header("Similar Players Analysis")

    # Target player selection
    player_options = profiles.apply(get_player_display_name, axis=1).tolist()
    player_ids = profiles["player_id"].tolist()

    target_idx = st.selectbox(
        "Target Player",
        options=range(len(player_options)),
        format_func=lambda i: player_options[i],
        key="similar_target",
    )
    target_id = player_ids[target_idx]

    # Configuration
    with st.expander("Similarity Settings"):
        col1, col2 = st.columns(2)
        with col1:
            min_minutes = st.number_input("Min minutes for candidates", 0, 3000, 450, 90)
            top_k = st.slider("Number of similar players", 1, 10, 5)
        with col2:
            metric = st.radio(
                "Distance metric",
                options=["cosine", "euclidean"],
                format_func=lambda x: "Cosine Similarity" if x == "cosine" else "Euclidean Distance",
            )

    # Compute similar players
    config = ScoutingConfig(
        min_minutes=min_minutes,
        similarity_top_k=top_k,
        similarity_metric=metric,
    )

    with st.spinner("Finding similar players..."):
        similar = compute_similar_players(profiles, target_id, config)

    target_name = profiles.loc[profiles["player_id"] == target_id, "player_name"].values[0]

    if similar is None or similar.empty:
        show_empty_state(f"No similar players found for {target_name} with current settings.")
        return

    st.subheader(f"Players Similar to {target_name}")

    # Display table
    display_df = similar[["player_name", "team_name", "position_band", "similarity", "minutes", "feature_contributions"]].copy()
    display_df["similarity"] = display_df["similarity"].apply(lambda x: f"{x:.3f}")
    display_df["feature_contributions"] = display_df["feature_contributions"].apply(
        lambda d: ", ".join(f"{k}: {v:.2f}" for k, v in list(d.items())[:3])
    )
    display_df = display_df.rename(columns={
        "player_name": "Player",
        "team_name": "Team",
        "position_band": "Band",
        "similarity": "Similarity",
        "minutes": "Min",
        "feature_contributions": "Top Feature Differences",
    })
    st.dataframe(display_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # Radar chart comparison
    st.subheader("Profile Comparison")
    comparison_players = profiles[profiles["player_id"].isin([target_id] + similar["player_id"].tolist())].copy()
    fig = create_radar_chart(comparison_players, title=f"Similarity to {target_name}")
    st.plotly_chart(fig, use_container_width=True)

    # Bar chart
    metric = st.selectbox(
        "Metric for bar comparison",
        options=[m[0] for m in get_comparison_metrics() if m[0].endswith("_per_90")],
        format_func=lambda x: dict(get_comparison_metrics()).get(x, x),
        key="similar_bar_metric",
    )
    fig = create_bar_comparison(
        comparison_players,
        metric,
        title=f"{dict(get_comparison_metrics()).get(metric, metric)} Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render_scouting_page()