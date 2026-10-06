"""
Simple save/list/get functions -- no generic ORM abstraction layer,
since this project has exactly one entity (Investigation) to persist.
"""

from ml_detective.db.models import Investigation
from ml_detective.db.session import get_session, init_db
from ml_detective.ingestion.findings import Severity
from ml_detective.reasoning.models import DetectiveFinding
from ml_detective.scoring.models import HealthScoreReport


def save_investigation(
    dataset_name: str,
    n_rows: int,
    n_cols: int,
    target_column: str | None,
    task_type: str,
    health_report: HealthScoreReport,
    detective_findings: list[DetectiveFinding],
    report: dict,
) -> int:
    init_db()
    session = get_session()
    try:
        record = Investigation(
            dataset_name=dataset_name,
            n_rows=n_rows,
            n_cols=n_cols,
            target_column=target_column,
            task_type=task_type,
            health_score=health_report.overall_score,
            grade=health_report.grade,
            findings_count=len(detective_findings),
            report_json={
                "findings": [f.to_dict() for f in detective_findings],
                "report": report,
            },
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record.id
    finally:
        session.close()


def list_investigations(limit: int = 20) -> list[Investigation]:
    init_db()
    session = get_session()
    try:
        return (
            session.query(Investigation)
            .order_by(Investigation.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()


def get_investigation(investigation_id: int) -> Investigation | None:
    init_db()
    session = get_session()
    try:
        return session.get(Investigation, investigation_id)
    finally:
        session.close()