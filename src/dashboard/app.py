"""Main entry point for the Premier League Intelligence Dashboard."""

from __future__ import annotations

import streamlit as st

from src.dashboard.pages.overview import render_overview_page
from src.dashboard.pages.player_explorer import render_player_explorer_page
from src.dashboard.pages.scouting_comparisons import render_scouting_page
from src.dashboard.pages.match_analysis import render_match_analysis_page
from src.dashboard.pages.league_simulator import render_league_simulator_page
from src.dashboard.utils import apply_custom_css
from src.dashboard.data_loader import _check_data_exists


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
        st.caption("2015/16 Season Analysis")

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
        st.caption("StatsBomb Open Data")
        st.caption("Premier League 2015/16")
        st.caption("380 matches, 20 teams, 550 players")

        st.divider()

        # Links
        st.markdown("**Resources**")
        st.markdown("- [GitHub Repository](https://github.com/ganesh-r-k-7/premier-league-intelligence)")
        st.markdown("- [StatsBomb Open Data](https://github.com/statsbomb/open-data)")

    # Route to page
    if page == "📊 Overview":
        render_overview_page()
    elif page == "🔍 Player Explorer":
        render_player_explorer_page()
    elif page == "🎯 Scouting & Comparisons":
        render_scouting_page()
    elif page == "📈 Match Analysis":
        render_match_analysis_page()
    elif page == "🏆 League Simulator":
        render_league_simulator_page()


if __name__ == "__main__":
    main()