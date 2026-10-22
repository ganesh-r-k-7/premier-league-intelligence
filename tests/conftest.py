import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def competitions_fixture():
    return json.loads((FIXTURES_DIR / "competitions.json").read_text(encoding="utf-8"))


@pytest.fixture
def matches_fixture():
    return json.loads((FIXTURES_DIR / "matches_2_99.json").read_text(encoding="utf-8"))


@pytest.fixture
def events_synthetic():
    return json.loads((FIXTURES_DIR / "events_synthetic.json").read_text(encoding="utf-8"))


@pytest.fixture
def lineups_synthetic():
    return json.loads((FIXTURES_DIR / "lineups_synthetic.json").read_text(encoding="utf-8"))
