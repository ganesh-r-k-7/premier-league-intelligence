import json

import pandas
import plotly.express as px
from src.data import config

SOURCE_ATTRIBUTION = (
    "StatsBomb Open Data (https://github.com/statsbomb/open-data). "
    "Attribution to StatsBomb is required by the source terms."
)


def load_competitions():
    path = config.STATSBOMB_RAW_DIR / "competitions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_fixtures():
    return pandas.read_parquet(config.STATSBOMB_PROCESSED_DIR / "fixtures.parquet")


def _outcome_counts(fixtures):
    home_wins = draws = away_wins = 0
    goals = []
    for row in fixtures.itertuples(index=False):
        home_score = row.home_score
        away_score = row.away_score
        if home_score is None or away_score is None:
            continue
        goals.append(home_score + away_score)
        if home_score > away_score:
            home_wins += 1
        elif home_score < away_score:
            away_wins += 1
        else:
            draws += 1
    return home_wins, draws, away_wins, goals


def _points_table(fixtures):
    points = {}
    for row in fixtures.itertuples(index=False):
        home_score = row.home_score
        away_score = row.away_score
        if home_score is None or away_score is None:
            continue
        home_points = home_score * 3
        away_points = away_score * 3
        if home_score == away_score:
            home_points += 1
            away_points += 1
        points[row.home_team_name] = points.get(row.home_team_name, 0) + home_points
        points[row.away_team_name] = points.get(row.away_team_name, 0) + away_points
    return sorted(points.items(), key=lambda item: -item[1])


def _season_coverage(fixtures):
    counts = {}
    if "season_name" not in fixtures.columns:
        return counts
    for season in fixtures["season_name"]:
        if isinstance(season, str):
            counts[season] = counts.get(season, 0) + 1
    return counts


def describe(competitions, fixtures):
    competition_counts = {}
    for row in competitions:
        name = row.get("competition_name")
        if isinstance(name, str):
            competition_counts[name] = competition_counts.get(name, 0) + 1
    season_counts = {}
    for row in competitions:
        season = row.get("season")
        if isinstance(season, dict):
            season_id = season.get("season_id")
            if isinstance(season_id, int):
                season_counts[season_id] = season_counts.get(season_id, 0) + 1

    home_wins, draws, away_wins, goals = _outcome_counts(fixtures)
    mean_goals = (sum(goals) / len(goals)) if goals else 0.0
    date_values = fixtures["match_date"].dropna() if "match_date" in fixtures.columns else []
    date_range = (
        [str(min(date_values)), str(max(date_values))] if len(date_values) else [None, None]
    )
    return {
        "competitions": competition_counts,
        "seasons": season_counts,
        "fixtures": {
            "rows": len(fixtures),
            "home_team_count": len(set(fixtures["home_team_name"])),
            "away_team_count": len(set(fixtures["away_team_name"])),
            "date_range": date_range,
            "mean_total_goals": mean_goals,
            "home_wins": home_wins,
            "draws": draws,
            "away_wins": away_wins,
            "season_coverage": _season_coverage(fixtures),
            "top_teams_by_points": _points_table(fixtures)[:10],
            "sources": sorted(set(fixtures["source"])) if "source" in fixtures.columns else [],
        },
    }


def render_markdown(summary):
    fixtures = summary["fixtures"]
    lines = []
    lines.append("# Phase 1 exploratory analysis")
    lines.append("")
    lines.append(
        "This report summarises the data that is actually available after ingestion "
        "and validation. Every figure is computed from the ingested raw files; "
        "no model has been trained and no value is estimated."
    )
    lines.append("")
    lines.append("## Data sources")
    lines.append("")
    lines.append("- %s" % SOURCE_ATTRIBUTION)
    lines.append(
        "- Only the target competition (the Premier League) and its available seasons "
        "were ingested in this phase."
    )
    lines.append(
        "- Historical player-level event data was not ingested: per-match event files "
        "are 2-4 MB each and total gigabytes."
    )
    lines.append("")
    lines.append("## Competition and season coverage")
    lines.append("")
    lines.append(
        "- Number of competitions in the ingested competitions file: %d."
        % len(summary["competitions"])
    )
    available = ", ".join(
        "%s (%d matches)" % (name, count) for name, count in sorted(fixtures["season_coverage"].items())
    )
    lines.append("- Ingested PL seasons: %s." % (available or "none"))
    lines.append("")
    lines.append("## Fixtures")
    lines.append("")
    lines.append("- Match rows: %d." % fixtures["rows"])
    lines.append("- Date range: %s to %s." % (fixtures["date_range"][0], fixtures["date_range"][1]))
    lines.append("- Mean total goals per match: %.2f." % fixtures["mean_total_goals"])
    lines.append(
        "- Home wins: %d, draws: %d, away wins: %d."
        % (fixtures["home_wins"], fixtures["draws"], fixtures["away_wins"])
    )
    lines.append("- Top ten teams by league points: %s." % fixtures["top_teams_by_points"])
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- The Premier League is covered for two seasons only (2015/2016 and 2003/2004).")
    lines.append(
        "- Player-level statistics, xG and event data are not part of this phase, so "
        "player forecasting and market-value modelling have no data yet."
    )
    lines.append(
        "- Points totals span a single ingested season; teams do not appear in every "
        "season, so the table is illustrative, not a multi-season ranking."
    )
    return "\n".join(lines) + "\n"


def build_figures(fixtures):
    figure = px.histogram(
        fixtures,
        x="home_score",
        nbins=15,
        title="Home goals per match",
    )
    return [figure]


def write_figures(figures, directory):
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, figure in enumerate(figures):
        path = directory / ("figure_%d.html" % index)
        figure.write_html(str(path))
        paths.append(path)
    return paths


def write_report(summary, path=None):
    from pathlib import Path

    path = Path(path) if path is not None else config.REPORTS_DIR / "phase1_eda_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(summary), encoding="utf-8")
    return path


def build_notebook():
    import nbformat

    setup = "from src import eda"
    load = "competitions = eda.load_competitions()\n" "fixtures = eda.load_fixtures()"
    describe_cell = "summary = eda.describe(competitions, fixtures)"
    figures = "figures = eda.build_figures(fixtures)\nfigures[0].show()"
    conclusions = "print(eda.render_markdown(summary))"
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Phase 1 exploratory analysis\n"
            "\n"
            "This notebook explores the ingested StatsBomb data. It calls the same "
            "functions that produce reports/phase1_eda_report.md."
        ),
        nbformat.v4.new_code_cell(setup),
        nbformat.v4.new_code_cell(load),
        nbformat.v4.new_code_cell(describe_cell),
        nbformat.v4.new_code_cell(figures),
        nbformat.v4.new_code_cell(conclusions),
    ]
    return nbformat.v4.new_notebook(cells=cells)


def execute(notebook):
    from nbclient import NotebookClient

    client = NotebookClient(notebook, timeout=600)
    client.execute()
    return notebook


def main():
    competitions = load_competitions()
    fixtures = load_fixtures()
    summary = describe(competitions, fixtures)
    report_path = write_report(summary)
    figures = build_figures(fixtures)
    figure_paths = write_figures(figures, config.REPORTS_DIR / "figures")
    print("wrote report to %s" % report_path)
    for path in figure_paths:
        print("wrote figure to %s" % path)
    return report_path


if __name__ == "__main__":
    main()
