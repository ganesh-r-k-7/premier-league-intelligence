"""Streamlit review tool for Phase 8.5B player identity candidates.

Local-only interactive review tool for 1,363 player identity candidates.
Supports filtering, detail view, bulk actions, and Transfermarkt/Wikipedia search links.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from pathlib import Path
from urllib.parse import quote_plus

from src.data import config as data_config


REVIEW_FILE = data_config.DATA_PROCESSED_DIR / "review" / "player_identity_review_v2.csv"

# Status vocabulary
VALID_STATUSES = ["pending", "confirmed", "rejected", "ambiguous"]
CANDIDATE_TYPES = ["cross_provider", "same_provider_continuity", "unresolved"]


@st.cache_data
def load_review_data():
    """Load review data with caching."""
    if not REVIEW_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(REVIEW_FILE)


def save_review_data(df: pd.DataFrame):
    """Save review data to CSV."""
    df.to_csv(REVIEW_FILE, index=False)


def generate_transfermarkt_url(player_name: str, team: str = "") -> str:
    """Generate Transfermarkt search URL."""
    query = player_name
    if team:
        query += f" {team}"
    return f"https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query={quote_plus(query)}"


def generate_wikipedia_url(player_name: str) -> str:
    """Generate Wikipedia search URL."""
    return f"https://en.wikipedia.org/w/index.php?search={quote_plus(player_name)}"


def render_candidate_card(row: pd.Series, idx: int):
    """Render a single candidate as an expandable card."""
    with st.expander(f"{row['src_a_name']} ({row['src_a_source']}) ↔ {row['src_b_name']} ({row['src_b_source']}) — Evidence: {row['evidence_score']}", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Source A (Reference)**")
            st.write(f"**Name:** {row['src_a_name']}")
            st.write(f"**ID:** {row['src_a_id']}")
            st.write(f"**Source:** {row['src_a_source']}")
            st.write(f"**Team:** {row['src_a_team'] if pd.notna(row['src_a_team']) else 'N/A'}")
            st.write(f"**Season:** {row['src_a_season']}")
            st.write(f"**Position:** {row['src_a_position'] if pd.notna(row['src_a_position']) else 'N/A'}")
            if pd.notna(row.get('src_a_normalized_name')):
                st.write(f"**Normalized Name:** {row['src_a_normalized_name']}")

        with col2:
            st.markdown("**Source B (Candidate)**")
            st.write(f"**Name:** {row['src_b_name'] if pd.notna(row['src_b_name']) else 'N/A'}")
            st.write(f"**ID:** {row['src_b_id'] if pd.notna(row['src_b_id']) else 'N/A'}")
            st.write(f"**Source:** {row['src_b_source'] if pd.notna(row['src_b_source']) else 'N/A'}")
            st.write(f"**Team:** {row['src_b_team'] if pd.notna(row['src_b_team']) else 'N/A'}")
            st.write(f"**Season:** {row['src_b_season'] if pd.notna(row['src_b_season']) else 'N/A'}")
            st.write(f"**Position:** {row['src_b_position'] if pd.notna(row['src_b_position']) else 'N/A'}")
            if pd.notna(row.get('src_b_normalized_name')):
                st.write(f"**Normalized Name:** {row['src_b_normalized_name']}")

        st.markdown("---")
        st.markdown("**Evidence**")
        ev_col1, ev_col2, ev_col3, ev_col3b = st.columns(4)
        with ev_col1:
            st.metric("Name Similarity", f"{row['name_similarity']:.2f}")
        with ev_col2:
            st.metric("Team Match", "✓" if row['team_match'] else "✗")
        with ev_col3:
            st.metric("Season Adjacent", "✓" if row['season_adjacent'] else "✗")
        with ev_col3b:
            st.metric("Position Match", "✓" if row['position_match'] else "✗")

        ev_col4, ev_col5 = st.columns(2)
        with ev_col4:
            st.metric("Evidence Score", f"{row['evidence_score']:.2f}")
        with ev_col5:
            st.metric("Type", row['candidate_type'].replace('_', ' ').title())

        # FPL ID reuse severity
        severity = row.get('fpl_id_reuse_severity')
        if pd.notna(severity):
            severity_colors = {
                "Critical": "🔴",
                "High": "🟠",
                "Medium": "🟡",
                "Low": "🟢",
                "None": "⚪"
            }
            st.warning(f"{severity_colors.get(severity, '')} FPL ID Reuse Severity: {severity}")

        # Additional evidence for same-provider
        if row['candidate_type'] == 'same_provider_continuity':
            st.markdown("**Same-Provider Details**")
            col_a, col_b = st.columns(2)
            with col_a:
                if pd.notna(row.get('minutes_delta')):
                    st.write(f"Minutes Delta: {row['minutes_delta']:+.0f}")
                if pd.notna(row.get('goals_delta')):
                    st.write(f"Goals Delta: {row['goals_delta']:+.0f}")
            with col_b:
                if pd.notna(row.get('club_tenure_overlap')):
                    st.write(f"Club Tenure Overlap: {'✓' if row['club_tenure_overlap'] else '✗'}")
                if pd.notna(row.get('transfer_known')):
                    st.write(f"Transfer Known: {'✓' if row['transfer_known'] else '✗'}")

        # Quality flags
        for src in ['src_a', 'src_b']:
            flag = row.get(f'{src}_quality_flag')
            if pd.notna(flag) and flag:
                st.warning(f"⚠️ Quality Flag ({src}): {flag}")

        # External references
        st.markdown("**External References**")
        ext_col1, ext_col2 = st.columns(2)
        with ext_col1:
            if pd.notna(row['src_a_name']):
                tm_url = generate_transfermarkt_url(row['src_a_name'], str(row['src_a_team']) if pd.notna(row['src_a_team']) else "")
                wiki_url = generate_wikipedia_url(row['src_a_name'])
                st.link_button("🔍 Transfermarkt (A)", tm_url, use_container_width=True)
                st.link_button("📖 Wikipedia (A)", wiki_url, use_container_width=True)
        with ext_col2:
            if pd.notna(row['src_b_name']):
                tm_url = generate_transfermarkt_url(row['src_b_name'], str(row['src_b_team']) if pd.notna(row['src_b_team']) else "")
                wiki_url = generate_wikipedia_url(row['src_b_name'])
                st.link_button("🔍 Transfermarkt (B)", tm_url, use_container_width=True)
                st.link_button("📖 Wikipedia (B)", wiki_url, use_container_width=True)

        # External reference fields
        ext_url = st.text_input(
            "External Reference URL",
            value=row.get('external_reference_url', ''),
            key=f"ext_url_{idx}",
            placeholder="Paste Transfermarkt/Wikipedia URL here..."
        )
        ext_notes = st.text_area(
            "External Notes",
            value=row.get('external_notes', ''),
            key=f"ext_notes_{idx}",
            placeholder="Notes from external lookup..."
        )

        # Status selector
        current_status = row['status'] if pd.notna(row['status']) else 'pending'
        new_status = st.selectbox(
            "Status",
            VALID_STATUSES,
            index=VALID_STATUSES.index(current_status) if current_status in VALID_STATUSES else 0,
            key=f"status_{idx}"
        )

        notes = st.text_area(
            "Reviewer Notes",
            value=row.get('reviewer_notes', ''),
            key=f"notes_{idx}",
            placeholder="Reasoning for decision..."
        )

        # Return updated values
        return {
            'status': new_status,
            'reviewer_notes': notes,
            'external_reference_url': ext_url,
            'external_notes': ext_notes,
        }


def main():
    st.set_page_config(
        page_title="Player Identity Review",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Player Identity Review Tool")
    st.caption("Phase 8.5B — Local-only review tool for player identity candidates")

    # Load data
    df = load_review_data()
    if df.empty:
        st.error(f"Review file not found: {REVIEW_FILE}")
        st.info("Run `python -m src.data.player_identity_curation generate --version 2` to generate the review file.")
        return

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")

        # Candidate type filter
        candidate_types = st.multiselect(
            "Candidate Type",
            options=CANDIDATE_TYPES,
            default=CANDIDATE_TYPES,
        )

        # Status filter
        statuses = st.multiselect(
            "Status",
            options=VALID_STATUSES,
            default=VALID_STATUSES,
        )

        # Evidence score range
        min_score, max_score = st.slider(
            "Evidence Score Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.1,
        )

        # FPL ID reuse severity
        severities = st.multiselect(
            "FPL ID Reuse Severity",
            options=["Critical", "High", "Medium", "Low", "None"],
            default=["Critical", "High", "Medium", "Low", "None"],
        )

        # Season filter
        all_seasons = sorted(set(
            list(df['src_a_season'].dropna().astype(int).unique()) +
            list(df['src_b_season'].dropna().astype(int).unique())
        ))
        selected_seasons = st.multiselect(
            "Season",
            options=all_seasons,
            default=all_seasons,
        )

        # Team filter
        all_teams = sorted(set(
            list(df['src_a_team'].dropna().unique()) +
            list(df['src_b_team'].dropna().unique())
        ))
        selected_teams = st.multiselect(
            "Team",
            options=all_teams,
            default=all_teams,
        )

    # Apply filters
    filtered = df.copy()

    if candidate_types:
        filtered = filtered[filtered['candidate_type'].isin(candidate_types)]
    if statuses:
        filtered = filtered[filtered['status'].isin(statuses)]
    filtered = filtered[(filtered['evidence_score'] >= min_score) & (filtered['evidence_score'] <= max_score)]

    if severities:
        filtered = filtered[
            filtered['fpl_id_reuse_severity'].fillna('None').isin(severities)
        ]

    if selected_seasons:
        filtered = filtered[
            filtered['src_a_season'].isin(selected_seasons) |
            filtered['src_b_season'].isin(selected_seasons)
        ]

    if selected_teams:
        filtered = filtered[
            filtered['src_a_team'].isin(selected_teams) |
            filtered['src_b_team'].isin(selected_teams)
        ]

    # Overview stats
    st.markdown("### 📊 Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Candidates", len(df))
    with col2:
        st.metric("Filtered", len(filtered))
    with col3:
        st.metric("Confirmed", len(df[df['status'] == 'confirmed']))
    with col4:
        st.metric("Rejected", len(df[df['status'] == 'rejected']))
    with col5:
        st.metric("Pending", len(df[df['status'] == 'pending']))

    # Status breakdown
    st.markdown("### 📈 Status Breakdown")
    status_counts = df['status'].value_counts()
    st.bar_chart(status_counts)

    # Candidate type breakdown
    st.markdown("### 📋 Candidate Type Breakdown")
    type_counts = df['candidate_type'].value_counts()
    st.bar_chart(type_counts)

    # Evidence score distribution
    st.markdown("### 📊 Evidence Score Distribution")
    hist_data = df['evidence_score'].value_counts().sort_index()
    st.bar_chart(hist_data)

    # FPL ID reuse severity distribution
    st.markdown("### ⚠️ FPL ID Reuse Severity")
    severity_counts = df['fpl_id_reuse_severity'].fillna('None').value_counts()
    st.bar_chart(severity_counts)

    # Export buttons
    st.markdown("### 💾 Export")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        if st.button("💾 Save Changes to CSV"):
            save_review_data(df)
            st.success("Changes saved to CSV!")
    with col_exp2:
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Review CSV",
            data=csv,
            file_name="player_identity_review_v2.csv",
            mime="text/csv"
        )
    with col_exp3:
        if st.button("🔗 Generate Canonical Mapping"):
            generate_canonical_mapping(df)

    # Candidate list
    st.markdown("---")
    st.markdown(f"### 👥 Candidates ({len(filtered)} of {len(df)})")

    # Pagination
    page_size = 20
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)

    if 'page' not in st.session_state:
        st.session_state.page = 1

    page = st.number_input(
        "Page",
        min_value=1,
        max_value=total_pages,
        value=min(st.session_state.page, total_pages),
        step=1,
        key="page_input"
    )
    st.session_state.page = page

    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered))
    page_df = filtered.iloc[start_idx:end_idx].reset_index(drop=True)

    st.caption(f"Showing {start_idx + 1}–{end_idx} of {len(filtered)} candidates")

    # Render candidates
    for idx, row in page_df.iterrows():
        updated = render_candidate_card(row, idx)

        # Check for changes
        if (updated['status'] != row['status'] or
            updated['reviewer_notes'] != row.get('reviewer_notes', '') or
            updated['external_reference_url'] != row.get('external_reference_url', '') or
            updated['external_notes'] != row.get('external_notes', '')):

            # Update the dataframe
            df.loc[idx, 'status'] = updated['status']
            df.loc[idx, 'reviewer_notes'] = updated['reviewer_notes']
            df.loc[idx, 'external_reference_url'] = updated['external_reference_url']
            df.loc[idx, 'external_notes'] = updated['external_notes']

            # Auto-save timestamp
            from datetime import datetime
            if updated['status'] != 'pending':
                df.loc[idx, 'reviewed_by'] = 'reviewer'  # Could be made configurable
                df.loc[idx, 'reviewed_at'] = datetime.now().isoformat()

            # Auto-save
            save_review_data(df)


def generate_canonical_mapping(df: pd.DataFrame):
    """Generate canonical identity mapping from confirmed candidates."""
    confirmed = df[df['status'] == 'confirmed'].copy()

    if confirmed.empty:
        st.warning("No confirmed mappings to apply.")
        return

    # Build canonical identity mapping
    canonical_rows = []
    for _, row in confirmed.iterrows():
        key = f"{row['src_a_source']}|{row['src_a_id']}|{row['src_a_season']}|{row['src_b_source']}|{row['src_b_id']}|{row['src_b_season']}"
        import hashlib
        canonical_id = int(hashlib.md5(key.encode()).hexdigest()[:12], 16)

        canonical_rows.append({
            "canonical_id": canonical_id,
            "source_a": row["src_a_source"],
            "source_a_id": row["src_a_id"],
            "source_a_name": row["src_a_name"],
            "source_a_season": row["src_a_season"],
            "source_a_team": row["src_a_team"],
            "source_b": row["src_b_source"],
            "source_b_id": row["src_b_id"],
            "source_b_name": row["src_b_name"],
            "source_b_season": row["src_b_season"],
            "source_b_team": row["src_b_team"],
            "evidence_score": row["evidence_score"],
            "candidate_type": row["candidate_type"],
            "reviewer": row.get("reviewed_by", ""),
            "reviewed_at": row.get("reviewed_at", ""),
            "notes": row.get("reviewer_notes", ""),
            "external_reference_url": row.get("external_reference_url", ""),
        })

    canonical_df = pd.DataFrame(canonical_rows)

    # Save
    out_path = data_config.DATA_PROCESSED_DIR / "canonical_identity_mapping.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_df.to_parquet(out_path, index=False)
    st.success(f"Generated canonical mapping: {out_path} ({len(canonical_df)} mappings)")


if __name__ == "__main__":
    main()