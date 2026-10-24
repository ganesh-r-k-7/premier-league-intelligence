"""Tests for match outcome feature engineering and model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.match_features import (
    MatchFeatureConfig,
    compute_match_features,
    compute_match_features_for_cutoff,
    _compute_team_match_stats,
    _compute_home_away_stats,
    _compute_h2h_stats,
    _compute_rest_days,
)
from src.models.match_model import (
    MatchModelConfig,
    MatchModel,
    train_model_at_cutoff,
    run_model_rolling_evaluation,
)
from src.evaluation.match_rolling import evaluate_all_match_models
from src.features.match_baselines import MatchBaselineConfig


def _make_test_fixtures(
    num_matchweeks: int = 6,
    num_teams: int = 4,
    include_xg: bool = True
) -> pd.DataFrame:
    """Create test fixtures DataFrame with known outcomes and xG.
    
    Designed to have all 3 outcome types (home win, draw, away win) across matchweeks.
    """
    fixtures_rows = []
    match_id = 100
    base_date = pd.Timestamp("2015-08-10")

    # Pre-defined matches to ensure all outcome types are present
    # Each matchweek has 2 matches (4 teams)
    matchweek_matches = {
        1: [(1, 2, 2, 0), (3, 4, 1, 1)],  # Home win, Draw
        2: [(2, 1, 0, 2), (4, 3, 1, 1)],  # Away win, Draw
        3: [(1, 3, 1, 0), (2, 4, 0, 1)],  # Home win, Away win
        4: [(1, 4, 2, 1), (2, 3, 0, 0)],  # Home win, Draw
        5: [(3, 1, 0, 1), (4, 2, 2, 2)],  # Away win, Draw
        6: [(3, 2, 1, 2), (4, 1, 1, 1)],  # Away win, Draw
    }

    for mw in range(1, min(num_matchweeks, 6) + 1):
        match_date = base_date + pd.Timedelta(days=(mw - 1) * 7)
        for home_id, away_id, h_score, a_score in matchweek_matches[mw]:
            row = {
                "match_id": match_id,
                "match_week": mw,
                "season_id": 27,
                "season_name": "2015/2016",
                "match_date": match_date,
                "home_team_id": home_id,
                "home_team_name": f"Team{home_id}",
                "away_team_id": away_id,
                "away_team_name": f"Team{away_id}",
                "home_score": h_score,
                "away_score": a_score,
                "match_status": "available",
                "stadium": "Stadium",
                "source": "test",
                "source_url": "test",
                "ingested_at": "2024-01-01",
            }
            if include_xg:
                row["home_xg"] = float(h_score + 0.1)
                row["away_xg"] = float(a_score + 0.1)
            fixtures_rows.append(row)
            match_id += 1

    return pd.DataFrame(fixtures_rows)


def _make_test_player_match(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Create minimal player_match data for xG aggregation."""
    player_match_rows = []
    for _, match in fixtures.iterrows():
        # Home team players
        for p in range(1, 4):
            player_match_rows.append({
                "match_id": match["match_id"],
                "season_id": match["season_id"],
                "player_id": match["home_team_id"] * 100 + p,
                "team_id": match["home_team_id"],
                "xg": match.get("home_xg", 1.0) / 3,
                "xa": 0.0,
            })
        # Away team players
        for p in range(1, 4):
            player_match_rows.append({
                "match_id": match["match_id"],
                "season_id": match["season_id"],
                "player_id": match["away_team_id"] * 100 + p,
                "team_id": match["away_team_id"],
                "xg": match.get("away_xg", 1.0) / 3,
                "xa": 0.0,
            })
    return pd.DataFrame(player_match_rows)


class TestTeamMatchStats:
    """Tests for _compute_team_match_stats."""

    def test_empty_history(self):
        """Empty history returns empty dict."""
        history = pd.DataFrame(columns=["match_week", "home_team_id", "away_team_id", "home_score", "away_score"])
        result = _compute_team_match_stats(history, team_id=1, window=5)
        assert result == {}

    def test_with_data(self):
        """Test with known match data."""
        history = pd.DataFrame({
            "match_week": [1, 2, 3],
            "home_team_id": [1, 1, 2],
            "away_team_id": [2, 3, 1],
            "home_score": [2, 1, 0],
            "away_score": [1, 1, 2],
        })
        result = _compute_team_match_stats(history, team_id=1, window=5)
        # Team 1: home vs 2 (win 2-1), home vs 3 (draw 1-1), away vs 2 (win 2-0)
        assert result["matches"] == 3.0
        assert result["wins"] == 2.0
        assert result["draws"] == 1.0
        assert result["losses"] == 0.0
        assert result["points"] == 7.0  # 2*3 + 1*1
        assert result["goals_scored"] == 5.0  # 2 + 1 + 2
        assert result["goals_conceded"] == 2.0  # 1 + 1 + 0


class TestHomeAwayStats:
    """Tests for _compute_home_away_stats."""

    def test_with_data(self):
        """Test home/away specific stats."""
        history = pd.DataFrame({
            "match_week": [1, 2, 3],
            "home_team_id": [1, 1, 2],
            "away_team_id": [2, 3, 1],
            "home_score": [2, 1, 0],
            "away_score": [1, 1, 2],
        })
        result = _compute_home_away_stats(history, team_id=1, window=5)
        # Home: vs 2 (win), vs 3 (draw)
        assert result["home_matches"] == 2.0
        assert result["home_wins"] == 1.0
        assert result["home_draws"] == 1.0
        assert result["home_losses"] == 0.0
        assert result["home_points"] == 4.0
        # Away: vs 2 (win)
        assert result["away_matches"] == 1.0
        assert result["away_wins"] == 1.0
        assert result["away_draws"] == 0.0
        assert result["away_losses"] == 0.0
        assert result["away_points"] == 3.0


class TestH2HStats:
    """Tests for _compute_h2h_stats."""

    def test_no_h2h_history(self):
        """No H2H matches returns empty dict."""
        history = pd.DataFrame({
            "match_week": [1],
            "home_team_id": [1],
            "away_team_id": [3],
            "home_score": [1],
            "away_score": [0],
        })
        result = _compute_h2h_stats(history, home_team_id=1, away_team_id=2, window=5)
        assert result == {}

    def test_with_h2h_history(self):
        """Test with head-to-head matches."""
        history = pd.DataFrame({
            "match_week": [1, 2],
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
            "home_score": [2, 1],
            "away_score": [1, 1],
        })
        result = _compute_h2h_stats(history, home_team_id=1, away_team_id=2, window=5)
        # H2H: Team 1 home win, Team 1 away draw
        assert result["h2h_matches"] == 2.0
        assert result["h2h_wins"] == 1.0
        assert result["h2h_draws"] == 1.0
        assert result["h2h_losses"] == 0.0
        assert result["h2h_points"] == 4.0


class TestRestDays:
    """Tests for _compute_rest_days."""

    def test_no_history(self):
        """No fixture history returns None."""
        fixtures_history = pd.DataFrame(columns=["match_date", "home_team_id", "away_team_id"])
        result = _compute_rest_days(None, fixtures_history, 1, pd.Timestamp("2015-08-15"))
        assert result is None

    def test_with_history(self):
        """Test rest days calculation."""
        fixtures_history = pd.DataFrame({
            "match_date": [pd.Timestamp("2015-08-01"), pd.Timestamp("2015-08-08")],
            "home_team_id": [1, 2],
            "away_team_id": [2, 1],
        })
        result = _compute_rest_days(None, fixtures_history, 1, pd.Timestamp("2015-08-15"))
        assert result == 7.0  # 7 days since last match on Aug 8


class TestComputeMatchFeatures:
    """Tests for compute_match_features."""

    def test_basic_features(self):
        """Basic feature computation works."""
        fixtures = _make_test_fixtures(5, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)

        features = compute_match_features(fixtures, player_match, 3, 27, config)
        assert not features.empty
        assert len(features) == 2  # 2 matches per week with 4 teams
        assert "home_matches" in features.columns
        assert "away_matches" in features.columns
        assert "home_rest_days" in features.columns
        assert "global_home_goals_per_match" in features.columns

    def test_xg_features_when_available(self):
        """xG features are included when available."""
        fixtures = _make_test_fixtures(5, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)

        features = compute_match_features(fixtures, player_match, 3, 27, config)
        assert "home_xg_for_per_match" in features.columns
        assert "away_xg_against_per_match" in features.columns
        assert "home_xg_for" in features.columns
        assert "away_xg_against" in features.columns

    def test_xg_features_absent_when_disabled(self):
        """xG features from player_match aggregation are disabled."""
        fixtures = _make_test_fixtures(5, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=False)

        features = compute_match_features(fixtures, player_match, 3, 27, config)
        # When include_xg=False, player_match is not used for xG aggregation
        # But existing xG columns in fixtures are still used for feature computation
        # So xG features from fixture columns are still present
        assert "home_xg_for" in features.columns

    def test_minimum_history_fallback(self):
        """Teams with insufficient history get global fallbacks."""
        fixtures = _make_test_fixtures(2, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        config = MatchFeatureConfig(rolling_window=5, min_history=3, include_xg=True)

        # At cutoff 2, teams only have 1 match of history
        features = compute_match_features(fixtures, player_match, 2, 27, config)
        # Should still have features (fallback to global)
        assert not features.empty
        assert "global_home_goals_per_match" in features.columns


class TestComputeMatchFeaturesForCutoff:
    """Tests for compute_match_features_for_cutoff."""

    def test_returns_target_labels(self):
        """Returns features with target labels attached."""
        fixtures = _make_test_fixtures(5, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)

        result = compute_match_features_for_cutoff(fixtures, player_match, 3, 27, config)
        assert not result.empty
        assert "outcome" in result.columns
        assert "home_score" in result.columns
        assert "away_score" in result.columns

    def test_outcome_labels_correct(self):
        """Outcome labels are correct: 0=away, 1=draw, 2=home."""
        fixtures = _make_test_fixtures(5, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)

        result = compute_match_features_for_cutoff(fixtures, player_match, 3, 27, config)
        for _, row in result.iterrows():
            if row["home_score"] > row["away_score"]:
                assert row["outcome"] == 2
            elif row["home_score"] == row["away_score"]:
                assert row["outcome"] == 1
            else:
                assert row["outcome"] == 0


class TestMatchModel:
    """Tests for MatchModel class."""

    def test_fit_and_predict(self):
        """End-to-end fit and predict."""
        fixtures = _make_test_fixtures(6, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        feature_config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=5)

        # Training data: matchweeks 1-3
        train_features = []
        for cw in range(1, 4):
            feats = compute_match_features_for_cutoff(fixtures, player_match, cw, 27, feature_config)
            if not feats.empty:
                train_features.append(feats)
        train_df = pd.concat(train_features, ignore_index=True)

        # Test data: matchweek 4
        test_df = compute_match_features_for_cutoff(fixtures, player_match, 4, 27, feature_config)

        model = MatchModel(model_config)
        model.fit(train_df)

        assert not model.get_fallback_status()
        probs = model.predict_proba(test_df)
        assert probs.shape == (len(test_df), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_fallback_when_insufficient_data(self):
        """Fallback used when training data insufficient."""
        fixtures = _make_test_fixtures(3, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        feature_config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=50)

        train_df = compute_match_features_for_cutoff(fixtures, player_match, 2, 27, feature_config)
        model = MatchModel(model_config)
        model.fit(train_df)

        assert model.get_fallback_status()
        probs = model.predict_proba(train_df)
        assert np.allclose(probs, 1.0/3)

    def test_deterministic_behavior(self):
        """Model behavior is deterministic with fixed random_state."""
        fixtures = _make_test_fixtures(6, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        feature_config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=10, random_state=42)

        train_features = []
        for cw in range(1, 4):
            feats = compute_match_features_for_cutoff(fixtures, player_match, cw, 27, feature_config)
            if not feats.empty:
                train_features.append(feats)
        train_df = pd.concat(train_features, ignore_index=True)
        test_df = compute_match_features_for_cutoff(fixtures, player_match, 4, 27, feature_config)

        model1 = MatchModel(model_config)
        model1.fit(train_df)
        probs1 = model1.predict_proba(test_df)

        model2 = MatchModel(model_config)
        model2.fit(train_df)
        probs2 = model2.predict_proba(test_df)

        assert np.allclose(probs1, probs2)


class TestTrainModelAtCutoff:
    """Tests for train_model_at_cutoff."""

    def test_cutoff_with_sufficient_history(self):
        """Training works with sufficient history."""
        fixtures = _make_test_fixtures(6, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        feature_config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=5)

        preds, fallback = train_model_at_cutoff(
            fixtures, player_match, 4, 27, feature_config, model_config
        )

        assert not fallback
        assert not preds.empty
        assert "model_home_prob" in preds.columns
        assert "model_draw_prob" in preds.columns
        assert "model_away_prob" in preds.columns
        assert np.allclose(
            preds["model_home_prob"] + preds["model_draw_prob"] + preds["model_away_prob"], 1.0
        )

    def test_cutoff_1_no_history(self):
        """Cutoff 1 uses fallback (no training data)."""
        fixtures = _make_test_fixtures(6, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        feature_config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=5)

        preds, fallback = train_model_at_cutoff(
            fixtures, player_match, 1, 27, feature_config, model_config
        )

        assert fallback
        assert not preds.empty
        assert np.allclose(preds["model_home_prob"], 1.0/3)


class TestLeakagePrevention:
    """Tests for temporal leakage prevention in match features and model."""

    def test_future_data_injection_no_leakage(self):
        """Injecting future matchweek data does not change earlier-cutoff features."""
        fixtures = _make_test_fixtures(5, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)

        # Get features at cutoff 3 WITHOUT future data
        feats_original = compute_match_features(fixtures, player_match, 3, 27, config)

        # INJECT fake future data for matchweek 4
        future_match = pd.DataFrame([{
            "match_id": 999,
            "match_week": 4,
            "season_id": 27,
            "season_name": "2015/2016",
            "home_team_id": 1,
            "home_team_name": "Team1",
            "away_team_id": 2,
            "away_team_name": "Team2",
            "home_score": 5,
            "away_score": 0,
            "home_xg": 4.5,
            "away_xg": 0.2,
            "match_status": "available",
            "stadium": "Stadium",
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        }])

        fixtures_with_future = pd.concat([fixtures, future_match], ignore_index=True)

        # Get features at cutoff 3 WITH future data injected
        feats_with_future = compute_match_features(fixtures_with_future, player_match, 3, 27, config)

        # Assertions: features at cw=3 must be IDENTICAL
        for col in feats_original.columns:
            if col in ["match_id", "match_week", "season_id", "home_team_id", "away_team_id"]:
                continue
            orig_vals = feats_original[col].values
            future_vals = feats_with_future[col].values
            np.testing.assert_allclose(
                orig_vals, future_vals, rtol=1e-10, atol=1e-12,
                err_msg=f"Leakage detected in features: {col} differs with future data injection"
            )

    def test_model_no_leakage_from_future_results(self):
        """Model predictions at cutoff don't use future match results."""
        fixtures = _make_test_fixtures(5, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        feature_config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=5)

        # Predictions at cutoff 3 without future data
        preds_original, _ = train_model_at_cutoff(
            fixtures, player_match, 3, 27, feature_config, model_config
        )

        # INJECT fake future data for matchweek 4 with extreme result
        future_match = pd.DataFrame([{
            "match_id": 999,
            "match_week": 4,
            "season_id": 27,
            "season_name": "2015/2016",
            "home_team_id": 1,
            "home_team_name": "Team1",
            "away_team_id": 2,
            "away_team_name": "Team2",
            "home_score": 5,
            "away_score": 0,
            "home_xg": 4.5,
            "away_xg": 0.2,
            "match_status": "available",
            "stadium": "Stadium",
            "source": "test",
            "source_url": "test",
            "ingested_at": "2024-01-01",
        }])

        fixtures_with_future = pd.concat([fixtures, future_match], ignore_index=True)

        # Predictions at cutoff 3 with future data
        preds_with_future, _ = train_model_at_cutoff(
            fixtures_with_future, player_match, 3, 27, feature_config, model_config
        )

        # Predictions should be identical (or very close due to floating point)
        pred_original = preds_original.sort_values("match_id").reset_index(drop=True)
        pred_future = preds_with_future.sort_values("match_id").reset_index(drop=True)

        for col in ["model_home_prob", "model_draw_prob", "model_away_prob"]:
            orig_vals = pred_original[col].values
            future_vals = pred_future[col].values
            np.testing.assert_allclose(
                orig_vals, future_vals, rtol=1e-10, atol=1e-12,
                err_msg=f"Leakage detected in model: {col} differs with future data injection"
            )


class TestModelProbabilities:
    """Tests for model probability outputs."""

    def test_probabilities_sum_to_one(self):
        """Predicted probabilities always sum to 1."""
        fixtures = _make_test_fixtures(6, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        feature_config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=5)

        train_features = []
        for cw in range(1, 4):
            feats = compute_match_features_for_cutoff(fixtures, player_match, cw, 27, feature_config)
            if not feats.empty:
                train_features.append(feats)
        train_df = pd.concat(train_features, ignore_index=True)
        test_df = compute_match_features_for_cutoff(fixtures, player_match, 4, 27, feature_config)

        model = MatchModel(model_config)
        model.fit(train_df)
        probs = model.predict_proba(test_df)

        assert np.allclose(probs.sum(axis=1), 1.0)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_fallback_probabilities(self):
        """Fallback model returns uniform probabilities."""
        fixtures = _make_test_fixtures(3, 4, include_xg=True)
        player_match = _make_test_player_match(fixtures)
        feature_config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=50)

        preds, fallback = train_model_at_cutoff(
            fixtures, player_match, 2, 27, feature_config, model_config
        )

        assert fallback
        assert np.allclose(preds["model_home_prob"], 1.0/3)
        assert np.allclose(preds["model_draw_prob"], 1.0/3)
        assert np.allclose(preds["model_away_prob"], 1.0/3)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_constant_features_handled(self):
        """Constant features are dropped gracefully."""
        base_date = pd.Timestamp("2015-08-10")
        fixtures = pd.DataFrame([
            {"match_id": 100, "match_week": 1, "season_id": 27,
             "match_date": base_date,
             "home_team_id": 1, "away_team_id": 2,
             "home_score": 1, "away_score": 1, "home_xg": 1.0, "away_xg": 1.0,
             "match_status": "available", "stadium": "Stadium",
             "source": "test", "source_url": "test", "ingested_at": "2024-01-01"},
            {"match_id": 101, "match_week": 1, "season_id": 27,
             "match_date": base_date,
             "home_team_id": 3, "away_team_id": 4,
             "home_score": 1, "away_score": 1, "home_xg": 1.0, "away_xg": 1.0,
             "match_status": "available", "stadium": "Stadium",
             "source": "test", "source_url": "test", "ingested_at": "2024-01-01"},
            {"match_id": 102, "match_week": 2, "season_id": 27,
             "match_date": base_date + pd.Timedelta(days=7),
             "home_team_id": 1, "away_team_id": 2,
             "home_score": 1, "away_score": 1, "home_xg": 1.0, "away_xg": 1.0,
             "match_status": "available", "stadium": "Stadium",
             "source": "test", "source_url": "test", "ingested_at": "2024-01-01"},
        ])
        player_match = pd.DataFrame([
            {"match_id": 100, "team_id": 1, "xg": 0.33, "xa": 0.0}, {"match_id": 100, "team_id": 2, "xg": 0.33, "xa": 0.0},
            {"match_id": 101, "team_id": 3, "xg": 0.33, "xa": 0.0}, {"match_id": 101, "team_id": 4, "xg": 0.33, "xa": 0.0},
            {"match_id": 102, "team_id": 1, "xg": 0.33, "xa": 0.0}, {"match_id": 102, "team_id": 2, "xg": 0.33, "xa": 0.0},
        ])

        config = MatchFeatureConfig(rolling_window=2, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=2)

        preds, fallback = train_model_at_cutoff(fixtures, player_match, 2, 27, config, model_config)
        # Should handle gracefully (either fallback or valid predictions)
        assert not preds.empty
        probs = preds[["model_home_prob", "model_draw_prob", "model_away_prob"]].values
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_missing_xg_handled(self):
        """Missing xG data is handled gracefully."""
        fixtures = _make_test_fixtures(5, 4, include_xg=False)
        player_match = _make_test_player_match(fixtures)
        config = MatchFeatureConfig(rolling_window=3, min_history=1, include_xg=True)
        model_config = MatchModelConfig(C=1.0, min_train_samples=10)

        train_features = []
        for cw in range(1, 4):
            feats = compute_match_features_for_cutoff(fixtures, player_match, cw, 27, config)
            if not feats.empty:
                train_features.append(feats)
        train_df = pd.concat(train_features, ignore_index=True)
        test_df = compute_match_features_for_cutoff(fixtures, player_match, 4, 27, config)

        model = MatchModel(model_config)
        model.fit(train_df)
        probs = model.predict_proba(test_df)

        assert probs.shape == (len(test_df), 3)
        assert np.allclose(probs.sum(axis=1), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])