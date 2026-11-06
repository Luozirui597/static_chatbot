"""Experiment-only CLI; fake by default, opt-in real provider transport."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.models import Base, ChatSession, Message
from experiments.general_tool_search.api_client import RealAPIClient, RealAPIConfig
from experiments.general_tool_search.engine import (
    FakeDecisionClient,
    FakeFinalClient,
    GeneralSearchEngine,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "general_cases.json"


def _seed(tmp_path: Path, *, case_name: str, question: str, hidden, visible):
    engine = create_engine(f"sqlite:///{tmp_path / 'cli.sqlite3'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    db = factory()
    session = ChatSession(title=case_name)
    db.add(session)
    db.flush()
    for content in list(hidden) + list(visible):
        db.add(
            Message(
                session_id=session.id,
                role="user",
                content=content,
                interaction_mode_snapshot="receive_teaching",
                prompt_version_snapshot="test",
                llm_profile_id_snapshot="fake",
                llm_profile_kind_snapshot="fake",
                llm_model_snapshot="fake",
            )
        )
    db.commit()
    current = Message(
        session_id=session.id,
        role="user",
        content=question,
        interaction_mode_snapshot="receive_teaching",
        prompt_version_snapshot="test",
        llm_profile_id_snapshot="fake",
        llm_profile_kind_snapshot="fake",
        llm_model_snapshot="fake",
    )
    db.add(current)
    db.commit()
    db.refresh(current)
    return engine, db, session.id, current.id



def write_evidence(directory: str | Path, *, case_name: str, result) -> Path:
    """Write a synthetic-only, human-checkable evidence JSON file.

    This intentionally excludes prompts, request bodies, decisions, headers,
    credentials, and raw provider errors.  It is only called for synthetic
    fixture runs.
    """
    if not getattr(result, "synthetic", False):
        raise ValueError("evidence export is synthetic-only")
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=False)
    search = result.search
    retrieved = []
    if search is not None:
        for item in search.results:
            retrieved.append(
                {
                    "message_id": item.get("message_id"),
                    "order": item.get("order"),
                    "content": item.get("content"),
                    "kind": item.get("kind"),
                    "truncated": bool(item.get("truncated", False)),
                }
            )
    payload = {
        "case": case_name,
        "synthetic": True,
        "final_reply": result.answer,
        "injected_source_ids": list(result.injected_source_ids),
        "retrieved_sources": retrieved,
        "cited_source_ids": list(result.cited_source_ids),
        "invalid_source_ids": list(result.invalid_source_ids),
        "search_limitations": {
            "truncated": bool(search.truncated) if search else False,
            "scan_truncated": bool(search.scan_truncated) if search else False,
            "corrections_truncated": bool(search.corrections_truncated) if search else False,
            "omitted_count": int(search.omitted_count) if search else 0,
            "pair_missing": bool(search.pair_missing) if search else False,
        },
    }
    path = target / "evidence.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--decision", default=None, help="fake JSON decision")
    parser.add_argument(
        "--answer",
        default="I cannot verify from the available evidence.",
        help="fake final answer",
    )
    parser.add_argument("--real-api", action="store_true")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--decision-timeout", type=float, default=None)
    parser.add_argument("--overall-timeout", type=float, default=None)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument(
        "--evidence-dir",
        default=None,
        help="synthetic-only evidence directory (must not exist)",
    )
    args = parser.parse_args()

    if args.real_api and args.evidence_dir:
        print("evidence export is synthetic-only; do not use it with --real-api")
        return 2

    real_config = None
    decision_timeout = 0.7
    overall_timeout = 5.0
    if args.real_api:
        if not args.api_key_env or not args.base_url or not args.model:
            print("real API requires --api-key-env, --base-url, and --model")
            return 2
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print("real API key environment variable is not set")
            return 2
        if args.decision_timeout is None or args.overall_timeout is None:
            print("real API requires --decision-timeout and --overall-timeout")
            return 2
        if args.decision_timeout <= 0 or args.overall_timeout <= 0:
            print("real API budgets must be positive")
            return 2
        decision_timeout = args.decision_timeout
        overall_timeout = args.overall_timeout
        real_config = RealAPIConfig(
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
            request_timeout_seconds=args.request_timeout,
            decision_json_mode=not args.no_json_mode,
        )

    data = json.loads(FIXTURE.read_text())
    case = data["cases"][args.case]
    with tempfile.TemporaryDirectory(prefix="general-tool-search-") as tmp:
        engine, db, session_id, current_id = _seed(
            Path(tmp),
            case_name=args.case,
            question=case["question"],
            hidden=case["hidden"],
            visible=data["visible"],
        )
        try:
            if real_config is not None:
                async def run_real():
                    real_client = RealAPIClient(real_config)
                    try:
                        result = await GeneralSearchEngine(
                            db,
                            session_id=session_id,
                            current_message_id=current_id,
                            decision_client=real_client,
                            final_client=real_client,
                            database_path=engine.url.database,
                            final_system_prompt="test system",
                            decision_timeout=decision_timeout,
                            overall_timeout=overall_timeout,
                        ).run()
                        return result, list(real_client.events)
                    finally:
                        await real_client.aclose()

                result, real_events = asyncio.run(run_real())
            else:
                decision = (
                    FakeDecisionClient([args.decision])
                    if args.decision
                    else FakeDecisionClient([])
                )
                result = asyncio.run(
                    GeneralSearchEngine(
                        db,
                        session_id=session_id,
                        current_message_id=current_id,
                        decision_client=decision,
                        final_client=FakeFinalClient([args.answer]),
                        database_path=engine.url.database,
                        final_system_prompt="test system",
                    ).run()
                )
            payload = {
                "status": result.status,
                "synthetic": result.synthetic,
                "counts": result.counts.as_dict(),
                "retrieved_source_ids": result.retrieved_source_ids,
                "cited_source_ids": result.cited_source_ids,
                "invalid_source_ids": result.invalid_source_ids,
            }
            if real_config is not None:
                payload["transport_events"] = [
                    event.as_dict() for event in real_events
                ]
            if args.evidence_dir:
                evidence_path = write_evidence(
                    args.evidence_dir,
                    case_name=args.case,
                    result=result,
                )
                payload["evidence_path"] = str(evidence_path)
            print(json.dumps(payload, sort_keys=True))
            return 0 if result.status.startswith("completed_") else 1
        finally:
            db.close()
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
