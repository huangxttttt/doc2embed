from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.core.config import settings
from app.schemas.database import ConversationSession, ConversationTurn


def _conversation_dir() -> Path:
    conversation_dir = Path(settings.conversation_storage_dir)
    if not conversation_dir.is_absolute():
        conversation_dir = Path.cwd() / conversation_dir
    conversation_dir.mkdir(parents=True, exist_ok=True)
    return conversation_dir


def _conversation_path(conversation_id: str) -> Path:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", conversation_id).strip("._")
    if not safe_id:
        raise ValueError("Invalid conversation id.")
    return _conversation_dir() / f"{safe_id}.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def load_or_create_session(conversation_id: str | None) -> ConversationSession:
    session_id = conversation_id or str(uuid4())
    path = _conversation_path(session_id)
    if not path.is_file():
        now = _utc_now()
        return ConversationSession(
            conversation_id=session_id,
            memory_summary="",
            turns=[],
            created_at=now,
            updated_at=now,
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    return ConversationSession.model_validate(data)


def save_session(session: ConversationSession) -> None:
    path = _conversation_path(session.conversation_id)
    path.write_text(
        session.model_dump_json(
            by_alias=True,
            exclude_none=True,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_conversation_context(session: ConversationSession) -> dict[str, object]:
    recent_turns = session.turns[-max(settings.conversation_recent_turns, 0):]
    return {
        "title": session.title,
        "memorySummary": session.memory_summary,
        "recentTurns": [
            {
                "turnId": turn.turn_id,
                "query": turn.query,
                "retrievalQuery": turn.retrieval_query,
                "finalSummary": turn.final_summary,
                "createdAt": turn.created_at.isoformat(),
            }
            for turn in recent_turns
        ],
        "turnCount": len(session.turns),
    }


def append_turn(
    session: ConversationSession,
    query: str,
    retrieval_query: str,
    final_summary: str,
    document_ids: list[str],
) -> ConversationTurn:
    turn = ConversationTurn(
        turn_id=str(uuid4()),
        query=query,
        retrieval_query=retrieval_query,
        final_summary=final_summary,
        created_at=_utc_now(),
        document_ids=document_ids,
    )
    session.turns.append(turn)
    session.updated_at = turn.created_at
    return turn
