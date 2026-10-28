"""Main entry point for the Premier League Intelligence Dashboard."""

from __future__ import annotations

import streamlit as st

from src.dashboard.pages.overview import render_overview_page
from src.dashboard.pages.player_explorer import render_player_explorer_page
from src.dashboard.pages.scouting_comparisons import render_scouting_page
from src.dashboard.pages.match_analysis import render_match_analysis_page
from src.dashboard.pages.league_simulator import render_league_simulator_page
from src.dashboard.utils import apply_custom_css
from src.dashboard.data_loader import (
    _check_data_exists,
    get_available_sources,
    get_available_seasons,
)

SEASON_LABELS = {
    "2025/26": "2025/26 (OpenFootball partial, FPL complete)",
    "2024/25": "2024/25 (OpenFootball complete, FPL complete)",
    "2023/24": "2023/24 (OpenFootball complete, FPL complete)",
    "2022/23": "2022/23 (OpenFootball complete, FPL xG/xA)",
    "2021/22": "2021/22 (OpenFootball complete, FPL no xG)",
    "2020/21": "2020/21 (OpenFootball complete, FPL no xG)",
    "2019/20": "2019/20 (OpenFootball complete, FPL transitional)",
    "2018/19": "2018/19 (OpenFootball complete, FPL legacy)",
    "2017/18": "2017/18 (OpenFootball complete, FPL legacy)",
    "2016/17": "2016/17 (OpenFootball complete, FPL legacy)",
    "2015/16": "2015/16 (StatsBomb complete, OpenFootball complete)",
    "2014/15": "2014/15 (OpenFootball complete)",
    "2013/14": "2013/14 (OpenFootball complete)",
    "2012/13": "2012/13 (OpenFootball complete)",
    "2011/12": "2011/12 (OpenFootball complete)",
    "2010/11": "2010/11 (OpenFootball complete)",
}


def main() -> None:
    """Main dashboard application."""
    st.set_page_config(
        page_title="Premier League Intelligence",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_css()

    # Check data availability
    data_exists, missing = _check_data_exists()
    if not data_exists:
        st.error("⚠️ Required data files not found:")
        for f in missing:
            st.code(f)
        st.markdown("""
        Please run the data ingestion pipeline first:
        ```bash
        python -m src.data.ingest_players --offline
        ```
        Or see the README for full setup instructions.
        """)
        st.stop()

    # Sidebar navigation
    with st.sidebar:
        st.title("⚽ PL Intelligence")
        st.caption("Multi-Season Premier League Analysis")

        # Data source selector
        sources = get_available_sources()
        source_options = list(sources.keys())
        selected_source = st.selectbox(
            "Data Source",
            options=source_options,
            index=0,
            help="Select the data provider. StatsBomb has detailed event data for 2015/16. OpenFootball has match results for 2010/11–2025/26. FPL has player statistics for 2016/17–2025/26."
        )

        # Season selector
        available_seasons = get_available_seasons(selected_source)
        if available_seasons:
            default_idx = available_seasons.index("2015/16") if "2015/16" in available_seasons else 0
            season_options = [f"{s} {SEASON_LABELS.get(s, '')}".strip() for s in available_seasons]
            season_labels_to_season = {f"{s} {SEASON_LABELS.get(s, '')}".strip(): s for s in available_seasons}

            selected_season_label = st.selectbox(
                "Season",
                options=season_options,
                index=default_idx,
                help="Select the season to analyze. Data availability varies by source and season."
            )
            selected_season = season_labels_to_season[selected_season_label]
        else:
            selected_season = "2015/16"
            st.warning("No seasons available for this source.")

        st.divider()

        page = st.radio(
            "Navigation",
            options=[
                "📊 Overview",
                "🔍 Player Explorer",
                "🎯 Scouting & Comparisons",
                "📈 Match Analysis",
                "🏆 League Simulator",
            ],
            index=0,
        )

        st.divider()

        # Data info
        st.markdown("**Data Source**")
        source_info = get_available_sources()[selected_source]
        st.caption(source_info["description"])
        st.caption(f"Selected Season: {selected_season}")

        st.divider()

        # Links
        st.markdown("**Resources**")
        st.markdown("- [GitHub Repository](https://github.com/ganesh-r-k-7/premier-league-intelligence)")
        st.markdown("- [StatsBomb Open Data](https://github.com/statsbomb/open-data)")
        st.markdown("- [OpenFootball](https://github.com/openfootball/football.json)")
        st.markdown("- [FPL Data](https://github.com/vaastav/Fantasy-Premier-League)")

    # Route to page with selected source and season
    if page == "📊 Overview":
        render_overview_page(selected_source, selected_season)
    elif page == "🔍 Player Explorer":
        render_player_explorer_page(selected_source, selected_season)
    elif page == "🎯 Scouting & Comparisons":
        render_scouting_page(selected_source, selected_season)
    elif page == "📈 Match Analysis":
        render_match_analysis_page(selected_source, selected_season)
    elif page == "🏆 League Simulator":
        render_league_simulator_page(selected_source, selected_season)


if __name__ == "__main__":
    main()