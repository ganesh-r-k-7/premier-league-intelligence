import pandas as pd

from src import eda


def _fixtures_frame():
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "match_date": pd.to_datetime(["2018-08-11", "2018-08-12", "2018-08-13"]),
            "season_name": ["2018/2019", "2018/2019", "2018/2019"],
            "home_team_name": ["Alpha FC", "Beta FC", "Alpha FC"],
            "away_team_name": ["Beta FC", "Alpha FC", "Beta FC"],
            "home_score": [2, 0, 1],
            "away_score": [1, 2, 1],
            "source": ["StatsBomb Open Data"] * 3,
        }
    )


def test_describe_counts_outcomes():
    summary = eda.describe([], _fixtures_frame())
    fixtures = summary["fixtures"]
    assert fixtures["rows"] == 3
    assert fixtures["home_wins"] == 1
    assert fixtures["away_wins"] == 1
    assert fixtures["draws"] == 1
    assert fixtures["season_coverage"] == {"2018/2019": 3}


def test_describe_handles_empty_goals():
    frame = _fixtures_frame()
    frame["home_score"] = [None, None, None]
    frame["away_score"] = [None, None, None]
    summary = eda.describe([], frame)
    assert summary["fixtures"]["mean_total_goals"] == 0.0


def test_render_markdown_includes_attribution_and_counts():
    summary = eda.describe([], _fixtures_frame())
    text = eda.render_markdown(summary)
    assert "StatsBomb" in text
    assert "Match rows: 3" in text
    assert "2018/2019" in text


def test_write_report_creates_file(tmp_path):
    summary = eda.describe([], _fixtures_frame())
    path = eda.write_report(summary, tmp_path / "report.md")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("# Phase 1 exploratory analysis")


def test_build_figures_returns_single_figure():
    figures = eda.build_figures(_fixtures_frame())
    assert len(figures) == 1


def test_write_figures_creates_html(tmp_path):
    figures = eda.build_figures(_fixtures_frame())
    paths = eda.write_figures(figures, tmp_path / "figures")
    assert len(paths) == 1
    assert paths[0].exists()
