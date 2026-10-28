"""Player Explorer page for the Premier League Intelligence Dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import (
    load_player_profiles,
    get_team_names,
    get_position_bands,
    search_players,
    get_comparison_metrics,
    get_fpl_metric_availability,
)
from src.dashboard.utils import (
    apply_custom_css,
    format_number,
    format_percentile,
    create_bar_comparison,
    create_percentile_bar,
    show_data_warning,
    show_empty_state,
    get_player_display_name,
    COLORS,
    POSITION_BAND_COLORS,
)


def render_player_explorer_page(source: str = "StatsBomb", season_name: str = "2015/16") -> None:
    """Render the Player Explorer page."""
    apply_custom_css()

    st.title("🔍 Player Explorer")
    st.caption(f"Search, filter, and explore {source} player statistics for {season_name}")

    # Show data source warning
    if source == "StatsBomb":
        show_data_warning(
            "Player statistics are from the 2015/16 Premier League season only (StatsBomb Open Data). "
            "Per-90 metrics for players with low minutes may be unreliable and are flagged."
        )
    elif source == "FPL":
        fpl_info = get_fpl_metric_availability().get(season_name, {})
        note = fpl_info.get("note", "FPL player statistics")
        show_data_warning(
            f"Player statistics are from FPL gameweek data for {season_name}. "
            f"{note}. xG/xA available from 2022/23 onward. "
            "Per-90 metrics for players with low minutes may be unreliable and are flagged."
        )
    else:
        show_data_warning(
            f"{source} provides match results only — no player-level statistics available."
        )

    # Load data
    try:
        profiles = load_player_profiles(source=source, season_name=season_name)
    except Exception as e:
        st.error(f"Failed to load player data: {e}")
        return

    if profiles.empty:
        st.warning(f"No player data available for {source} {season_name}.")
        return

    # Sidebar filters
    with st.sidebar:
        st.header("Filters")

        # Search
        search_query = st.text_input("🔍 Search player", placeholder="Enter player name...")

        # Team filter
        teams = ["All"] + get_team_names(source=source, season_name=season_name)
        selected_team = st.selectbox("Team", teams)

        # Position filter
        bands = ["All"] + get_position_bands()
        selected_band = st.selectbox("Position Band", bands)

        # Minimum minutes
        max_minutes = int(profiles["minutes"].max()) if len(profiles) > 0 and "minutes" in profiles.columns else 3420
        min_minutes = st.slider(
            "Minimum Minutes",
            min_value=0,
            max_value=max_minutes,
            value=0,
            step=90,
        )

        # Minimum appearances
        max_apps = int(profiles["appearances"].max()) if len(profiles) > 0 and "appearances" in profiles.columns else 38
        min_apps = st.slider(
            "Minimum Appearances",
            min_value=0,
            max_value=max_apps,
            value=0,
            step=1,
        )

        # Show only regular starters
        regular_only = st.checkbox("Regular starters only (starts_rate > 0.5)", value=False)

    # Apply filters
    filtered = profiles.copy()

    if search_query:
        filtered = filtered[
            filtered["player_name"].str.contains(search_query, case=False, na=False)
        ]

    if selected_team != "All":
        filtered = filtered[filtered["team_name"] == selected_team]

    if selected_band != "All":
        filtered = filtered[filtered["position_band"] == selected_band]

    filtered = filtered[filtered["minutes"] >= min_minutes]
    filtered = filtered[filtered["appearances"] >= min_apps]

    if regular_only:
        filtered = filtered[filtered["starts_rate"] > 0.5]

    # Results summary
    st.subheader(f"Results: {len(filtered)} players")
    if filtered.empty:
        show_empty_state("No players match the current filters. Try adjusting your criteria.")
        return

    # Show FPL metric availability note
    if source == "FPL":
        fpl_info = get_fpl_metric_availability().get(season_name, {})
        if fpl_info.get("xG") or fpl_info.get("xA"):
            st.info(f"✅ {season_name}: xG/xA and starts data available")
        else:
            st.warning(f"⚠️ {season_name}: {fpl_info.get('note', 'Limited metrics')}")

    # Metric selection for table
    st.subheader("Display Metrics")
    available_metrics = get_comparison_metrics(source=source, season_name=season_name)
    metric_cols = [m[0] for m in available_metrics]

    default_display = [
        "player_name", "team_name", "position_band", "minutes", "appearances",
        "goals", "goals_per_90", "assists", "assists_per_90",
        "xg", "xg_per_90", "shots", "shots_per_90",
    ]
    display_cols = st.multiselect(
        "Select columns to display",
        options=metric_cols + ["player_name", "team_name", "position", "position_band", "minutes", "appearances", "starts", "starts_rate"],
        default=[c for c in default_display if c in filtered.columns],
    )

    if not display_cols:
        display_cols = ["player_name", "team_name", "position_band", "minutes", "goals", "assists"]

    # Display table
    display_df = filtered[display_cols].copy()
    display_df = display_df.sort_values(
        by=[c for c in ["goals_per_90", "goals", "minutes"] if c in display_df.columns],
        ascending=False,
        na_position="last",
    )

    # Format numeric columns
    for col in display_df.select_dtypes(include=["float64", "float32"]).columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

    st.dataframe(
        display_df.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "player_name": "Player",
            "team_name": "Team",
            "position": "Position",
            "position_band": "Band",
            "minutes": "Min",
            "appearances": "Apps",
            "starts": "Starts",
            "starts_rate": "Start %",
            "goals": "Goals",
            "goals_per_90": "G/90",
            "assists": "Assists",
            "assists_per_90": "A/90",
            "xg": "xG",
            "xg_per_90": "xG/90",
            "npxg": "npxG",
            "npxg_per_90": "npxG/90",
            "xa": "xA",
            "xa_per_90": "xA/90",
            "shots": "Shots",
            "shots_per_90": "Shots/90",
            "shots_on_target": "SoT",
            "shots_on_target_per_90": "SoT/90",
            "expected_goals": "xG (FPL)",
            "expected_assists": "xA (FPL)",
            "expected_goals_per_90": "xG/90 (FPL)",
            "expected_assists_per_90": "xA/90 (FPL)",
        },
    )

    st.divider()

    # Player detail view
    st.header("Player Detail")
    player_options = filtered.apply(get_player_display_name, axis=1).tolist()
    selected_idx = st.selectbox(
        "Select a player for detailed view",
        options=range(len(player_options)),
        format_func=lambda i: player_options[i],
    )

    if selected_idx is not None:
        player = filtered.iloc[selected_idx]
        render_player_detail(player, profiles)


def render_player_detail(player: pd.Series, all_profiles: pd.DataFrame) -> None:
    """Render detailed view for a single player."""
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Profile")
        st.markdown(f"**{player['player_name']}**")
        st.caption(f"{player['team_name']} • {player.get('position', 'N/A')} ({player.get('position_band', 'N/A')})")

        # Key stats
        m1, m2, m3 = st.columns(3)
        m1.metric("Minutes", format_number(player.get("minutes", 0)))
        m2.metric("Appearances", format_number(player.get("appearances", 0)))
        m3.metric("Starts", format_number(player.get("starts", 0)))

        m1, m2, m3 = st.columns(3)
        m1.metric("Goals", format_number(player.get("goals", 0)))
        m2.metric("Assists", format_number(player.get("assists", 0)))

        # xG/xA (different column names for different sources)
        xg_val = player.get("xg", player.get("expected_goals", 0))
        xa_val = player.get("xa", player.get("expected_assists", 0))
        m3.metric("xG", f"{xg_val:.2f}")

        # Per-90 rates
        st.markdown("**Per 90 Minutes**")
        p1, p2, p3 = st.columns(3)
        g90 = player.get("goals_per_90", 0)
        a90 = player.get("assists_per_90", 0)
        xg90 = player.get("xg_per_90", player.get("expected_goals_per_90", 0))
        p1.metric("Goals/90", f"{g90:.2f}")
        p2.metric("Assists/90", f"{a90:.2f}")
        p3.metric("xG/90", f"{xg90:.2f}")

        p1, p2, p3 = st.columns(3)
        s90 = player.get("shots_per_90", 0)
        sot90 = player.get("shots_on_target_per_90", 0)
        xa90 = player.get("xa_per_90", player.get("expected_assists_per_90", 0))
        p1.metric("Shots/90", f"{s90:.2f}")
        p2.metric("SoT/90", f"{sot90:.2f}")
        p3.metric("xA/90", f"{xa90:.2f}")

        # Efficiency metrics
        st.markdown("**Efficiency**")
        e1, e2 = st.columns(2)
        gs = player.get("goals_per_shot", 0)
        goxg = player.get("goals_over_xg", 0)
        e1.metric("Goals/Shot", f"{gs:.3f}")
        e2.metric("Goals/xG", f"{goxg:.2f}")

        # Flags
        if player.get("minutes_flag", "ok") != "ok":
            st.warning(f"⚠️ Low minutes: {player['minutes_flag']}")
        if player.get("appearances_flag", "ok") != "ok":
            st.warning(f"⚠️ Low appearances: {player['appearances_flag']}")

        # High per-90 warnings
        for col in ["goals_per_90", "assists_per_90", "xg_per_90", "shots_per_90",
                    "expected_goals_per_90", "expected_assists_per_90"]:
            flag_col = f"{col}_high_warning"
            if player.get(flag_col, False):
                st.info(f"ℹ️ High {col.replace('_', ' ')} (above 95th percentile)")

    with col2:
        st.subheader("Percentile Ranks (within position band)")
        band = player.get("position_band", "")
        if band:
            band_profiles = all_profiles[all_profiles["position_band"] == band]
        else:
            band_profiles = all_profiles

        percentile_metrics = [
            ("goals_per_90", "Goals/90"),
            ("assists_per_90", "Assists/90"),
            ("xg_per_90", "xG/90"),
            ("xa_per_90", "xA/90"),
            ("expected_goals_per_90", "xG/90 (FPL)"),
            ("expected_assists_per_90", "xA/90 (FPL)"),
            ("shots_per_90", "Shots/90"),
            ("shots_on_target_per_90", "SoT/90"),
            ("goals_per_shot", "Goals/Shot"),
            ("goals_over_xg", "Goals over xG"),
            ("minutes_per_match", "Min/Match"),
        ]

        for metric, label in percentile_metrics:
            if metric in all_profiles.columns and pd.notna(player.get(metric)):
                value = player[metric]
                ref_values = band_profiles[metric].dropna()
                if len(ref_values) > 0:
                    pct = (ref_values <= value).mean() * 100
                    color = POSITION_BAND_COLORS.get(band, COLORS["primary"])
                    fig = create_percentile_bar(pct, color)
                    st.plotly_chart(fig, use_container_width=True, key=f"pct_{metric}_{player.name}")
                    st.caption(f"{label}: {value:.2f} ({pct:.0f}th percentile)")

    st.divider()

    # Comparison with teammates
    st.subheader(f"Comparison: {player['player_name']} vs {player['team_name']} Teammates")
    team_players = all_profiles[
        (all_profiles["team_name"] == player["team_name"]) &
        (all_profiles["position_band"] == player["position_band"]) &
        (all_profiles["minutes"] >= 270)
    ].copy()

    if len(team_players) > 1:
        metric = st.selectbox(
            "Metric to compare",
            options=[m[0] for m in get_comparison_metrics()],
            format_func=lambda x: dict(get_comparison_metrics()).get(x, x),
            key=f"compare_{player.get('player_id', player.name)}",
        )
        fig = create_bar_comparison(
            team_players,
            metric,
            title=f"{metric.replace('_', ' ').title()} — {player['team_name']} {player['position_band']}s",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No teammates with sufficient minutes in the same position band for comparison.")


if __name__ == "__main__":
    render_player_explorer_page()