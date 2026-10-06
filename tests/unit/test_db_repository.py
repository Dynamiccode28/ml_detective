"""
test_db_repository.py

Uses a separate temp SQLite file (via monkeypatch) so tests never
touch the real dev database.
"""

import pytest

from ml_detective.ingestion.findings import Severity
from ml_detective.reasoning.models import DetectiveFinding
from ml_detective.scoring.models import DimensionScore, HealthScoreReport


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Points the db module at a fresh temp SQLite file for this test only."""
    import ml_detective.db.session as session_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test.db"
    test_engine = create_engine(f"sqlite:///{db_path}")
    test_session_factory = sessionmaker(bind=test_engine)

    monkeypatch.setattr(session_module, "_engine", test_engine)
    monkeypatch.setattr(session_module, "_SessionLocal", test_session_factory)
    yield


def _build_health_report():
    dims = {d: DimensionScore(d, 80.0, 1, 1, True) for d in
            ["completeness", "uniqueness", "consistency", "validity", "accuracy", "timeliness"]}
    return HealthScoreReport(overall_score=82.5, grade="B", dimension_scores=dims)


def test_save_and_retrieve_investigation(temp_db):
    from ml_detective.db.repository import get_investigation, save_investigation

    findings = [DetectiveFinding("f1", {}, "r1", Severity.HIGH, 0.9, "b1", "m1", "fix1")]
    report = {"executive_summary": "test summary", "generated_with_llm": False}

    investigation_id = save_investigation(
        dataset_name="test.csv", n_rows=100, n_cols=5, target_column="target",
        task_type="binary_classification", health_report=_build_health_report(),
        detective_findings=findings, report=report,
    )

    retrieved = get_investigation(investigation_id)
    assert retrieved is not None
    assert retrieved.dataset_name == "test.csv"
    assert retrieved.health_score == 82.5
    assert retrieved.report_json["report"]["executive_summary"] == "test summary"


def test_list_investigations_returns_most_recent_first(temp_db):
    from ml_detective.db.repository import list_investigations, save_investigation

    for name in ["first.csv", "second.csv", "third.csv"]:
        save_investigation(
            dataset_name=name, n_rows=10, n_cols=2, target_column=None,
            task_type="unknown", health_report=_build_health_report(),
            detective_findings=[], report={},
        )

    results = list_investigations()
    assert len(results) == 3
    assert results[0].dataset_name == "third.csv"  # most recent first