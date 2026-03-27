from http.client import IncompleteRead
import json
import logging
from pathlib import Path
import ssl
from urllib import error, request
from uuid import UUID

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import MetaData, Table, inspect, select, text
from sqlalchemy.exc import NoSuchTableError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db_session
from app.models import DocumentSegment
from app.schemas.database import (
    ConversationSummaryDocument,
    ConversationSummaryEnvelope,
    ConversationSummaryResponse,
    ConversationTurnDocument,
    DatasetRetrieveEnvelope,
    DatasetRetrieveRequest,
    DatasetRetrieveResponse,
    DocumentSegmentRead,
    RetrievedDocumentAnswer,
    RetrievedDocumentContent,
    TableInfo,
    TableRowsResponse,
)
from app.services.conversation_memory import (
    append_turn,
    build_conversation_context,
    load_or_create_session,
    save_session,
)
from app.services.document_export import export_document_content
from app.services.llm import (
    analyze_documents,
    expand_query_for_retrieval,
    generate_conversation_title,
    summarize_conversation_history,
    summarize_conversation_memory,
)

router = APIRouter()
logger = logging.getLogger("uvicorn.error")

DbSession = Annotated[Session, Depends(get_db_session)]


def _dataset_retrieve_error_response(status_code: int, msg: str) -> JSONResponse:
    logger.warning(
        "Dataset API request failed: status=%s msg=%s",
        status_code,
        msg,
    )
    return JSONResponse(
        status_code=status_code,
        content={
            "code": "-1",
            "status": status_code,
            "msg": msg,
            "data": {},
            "success": False,
        },
    )





def _prepare_dify_query(query: str, fallback_query: str, max_length: int = 250) -> str:
    normalized = " ".join((query or "").split())
    fallback = " ".join((fallback_query or "").split())
    effective = normalized or fallback
    if not effective:
        raise ValueError("Query is required and cannot be empty.")
    if len(effective) > max_length:
        return effective[:max_length].rstrip()
    return effective


@router.get("/ping")
def ping_database(session: DbSession) -> dict[str, str]:
    try:
        session.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    return {"status": "ok", "database": "postgresql"}


@router.get("/tables", response_model=list[TableInfo])
def list_tables(
    session: DbSession,
    schema: str = Query(default="public"),
) -> list[TableInfo]:
    try:
        inspector = inspect(session.get_bind())
        tables = []
        for table_name in inspector.get_table_names(schema=schema):
            columns = [
                column["name"]
                for column in inspector.get_columns(table_name, schema=schema)
            ]
            tables.append(
                TableInfo(
                    schema_name=schema,
                    name=table_name,
                    columns=columns,
                )
            )
        return tables
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"Unable to list tables: {exc}") from exc


@router.get("/tables/{table_name}/rows", response_model=TableRowsResponse)
def get_table_rows(
    table_name: str,
    session: DbSession,
    schema: str = Query(default="public"),
    limit: int = Query(default=20, ge=1, le=100),
) -> TableRowsResponse:
    metadata = MetaData()
    try:
        table = Table(
            table_name,
            metadata,
            schema=schema,
            autoload_with=session.get_bind(),
        )
        stmt = select(table).limit(limit)
        result = session.execute(stmt)
        rows = [dict(row._mapping) for row in result]
        return TableRowsResponse(
            table=table_name,
            schema_name=schema,
            limit=limit,
            rows=rows,
        )
    except NoSuchTableError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Table '{schema}.{table_name}' does not exist.",
        ) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=503, detail=f"Unable to query table: {exc}") from exc


@router.get(
    "/document-segments/by-document/{document_id}",
    response_model=list[DocumentSegmentRead],
)
def get_document_segments_by_document_id(
    document_id: UUID,
    session: DbSession,
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[DocumentSegmentRead]:
    try:
        stmt = (
            select(DocumentSegment)
            .where(DocumentSegment.document_id == document_id)
            .order_by(DocumentSegment.position.asc())
            .limit(limit)
        )
        segments = session.scalars(stmt).all()
        return [DocumentSegmentRead.model_validate(segment) for segment in segments]
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to query document segments: {exc}",
        ) from exc


@router.get("/exports/{file_name}")
def download_exported_document(file_name: str) -> FileResponse:
    export_dir = Path(settings.document_export_dir)
    if not export_dir.is_absolute():
        export_dir = Path.cwd() / export_dir
    file_path = (export_dir / file_name).resolve()

    try:
        file_path.relative_to(export_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid export file path.") from exc

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Export file not found.")

    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@router.post("/datasets/retrieve", response_model=DatasetRetrieveEnvelope)
def retrieve_dataset(
    payload: DatasetRetrieveRequest,
    session: DbSession,
) -> DatasetRetrieveEnvelope | JSONResponse:
    logger.info(
        "Dataset retrieve started: dataset_id=%s conversation_id=%s query=%s",
        payload.dataset_id,
        payload.conversation_id or "",
        payload.query,
    )
    if not settings.dify_dataset_api_key:
        return _dataset_retrieve_error_response(
            500,
            "Dify dataset API key is not configured.",
        )

    dataset_url = (
        f"{settings.dify_base_url}/v1/datasets/"
        f"{payload.dataset_id}/retrieve"
    )

    try:
        conversation_session = load_or_create_session(payload.conversation_id)
        logger.info(
            "Conversation session loaded: conversation_id=%s existing_turns=%s title_present=%s",
            conversation_session.conversation_id,
            len(conversation_session.turns),
            bool(conversation_session.title),
        )
        if not conversation_session.title and not conversation_session.turns:
            conversation_session.title = generate_conversation_title(payload.query)
            logger.info(
                "Generated new conversation title: conversation_id=%s title=%s",
                conversation_session.conversation_id,
                conversation_session.title,
            )
        conversation_context = build_conversation_context(conversation_session)
        retrieval_query = expand_query_for_retrieval(
            payload.query,
            conversation_context=conversation_context,
        )
        dify_query = _prepare_dify_query(retrieval_query, payload.query)
        logger.info(
            "Retrieval query prepared: conversation_id=%s retrieval_query=%s dify_query_length=%s recent_turns=%s",
            conversation_session.conversation_id,
            dify_query,
            len(dify_query),
            len(conversation_context.get("recentTurns", [])),
        )

        body = json.dumps({"query": dify_query}).encode("utf-8")
        req = request.Request(
            dataset_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.dify_dataset_api_key}",
            },
            method="POST",
        )
        logger.info(
            "Calling Dify dataset retrieve: url=%s timeout_seconds=%s verify_ssl=%s",
            dataset_url,
            settings.dify_timeout_seconds,
            settings.dify_verify_ssl,
        )
        ssl_context = (
            ssl.create_default_context()
            if settings.dify_verify_ssl
            else ssl._create_unverified_context()
        )
        with request.urlopen(
            req,
            timeout=settings.dify_timeout_seconds,
            context=ssl_context,
        ) as response:
            response_body = response.read().decode("utf-8")
            data = json.loads(response_body)
        logger.info(
            "Dify dataset retrieve completed: conversation_id=%s record_count=%s",
            conversation_session.conversation_id,
            len(data.get("records", [])),
        )

        ordered_document_ids: list[UUID] = []
        seen_document_ids: set[UUID] = set()
        document_name_map: dict[UUID, str | None] = {}

        for record in data.get("records", []):
            segment = record.get("segment") or {}
            document_id_value = segment.get("document_id")
            if not document_id_value:
                continue
            try:
                document_id = UUID(str(document_id_value))
            except ValueError:
                continue

            document_info = segment.get("document") or {}
            document_name = document_info.get("name")
            document_name_map.setdefault(document_id, document_name)
            if document_id not in seen_document_ids:
                seen_document_ids.add(document_id)
                ordered_document_ids.append(document_id)
        logger.info(
            "Document ids extracted from Dify response: conversation_id=%s unique_documents=%s",
            conversation_session.conversation_id,
            len(ordered_document_ids),
        )

        documents: list[RetrievedDocumentContent] = []
        if ordered_document_ids:
            stmt = (
                select(DocumentSegment)
                .where(DocumentSegment.document_id.in_(ordered_document_ids))
                .order_by(
                    DocumentSegment.document_id.asc(),
                    DocumentSegment.position.asc(),
                )
            )
            segments = session.scalars(stmt).all()

            content_map: dict[UUID, list[str]] = {document_id: [] for document_id in ordered_document_ids}
            for segment in segments:
                content_map.setdefault(segment.document_id, []).append(segment.content)

            documents = [
                RetrievedDocumentContent(
                    document_id=document_id,
                    document_name=document_name_map.get(document_id),
                    content="\n".join(content_map.get(document_id, [])),
                )
                for document_id in ordered_document_ids
                if content_map.get(document_id)
            ]
        logger.info(
            "Document contents assembled: conversation_id=%s documents_with_content=%s",
            conversation_session.conversation_id,
            len(documents),
        )

        analysis_by_id, final_summary = analyze_documents(
            query=payload.query,
            documents=[
                {
                    "documentId": str(document.document_id),
                    "documentName": document.document_name,
                    "content": document.content,
                }
                for document in documents
            ],
            conversation_context=conversation_context,
        )
        logger.info(
            "Document analysis completed: conversation_id=%s analyzed_documents=%s summary_length=%s",
            conversation_session.conversation_id,
            len(analysis_by_id),
            len(final_summary),
        )
        turn = append_turn(
            conversation_session,
            query=payload.query,
            retrieval_query=retrieval_query,
            final_summary=final_summary,
            document_ids=[str(document.document_id) for document in documents],
            document_metadata=[
                ConversationTurnDocument(
                    document_id=str(document.document_id),
                    document_name=document.document_name,
                )
                for document in documents
            ],
        )
        logger.info(
            "Conversation turn appended: conversation_id=%s turn_id=%s document_count=%s",
            conversation_session.conversation_id,
            turn.turn_id,
            len(documents),
        )
        conversation_session.memory_summary = summarize_conversation_memory(
            query=payload.query,
            existing_memory=conversation_session.memory_summary,
            recent_turns=conversation_context.get("recentTurns", []),
            latest_final_summary=final_summary,
        )
        save_session(conversation_session)
        logger.info(
            "Conversation session saved: conversation_id=%s total_turns=%s memory_summary_length=%s",
            conversation_session.conversation_id,
            len(conversation_session.turns),
            len(conversation_session.memory_summary),
        )

        documents = [
            RetrievedDocumentContent(
                document_id=document.document_id,
                document_name=document.document_name,
                content=document.content,
                analysis=analysis_by_id.get(str(document.document_id)),
                storage_path=storage_path,
                download_url=download_url,
            )
            for document, (storage_path, download_url) in (
                (
                    document,
                    export_document_content(
                        document_id=str(document.document_id),
                        document_name=document.document_name,
                        content=document.content,
                    ),
                )
                for document in documents
            )
        ]
        logger.info(
            "Document exports generated: conversation_id=%s exported_documents=%s",
            conversation_session.conversation_id,
            len(documents),
        )

        response = DatasetRetrieveResponse(
            query=payload.query,
            conversation_id=conversation_session.conversation_id,
            conversation_title=conversation_session.title or generate_conversation_title(payload.query),
            documents=[
                RetrievedDocumentAnswer(
                    analysis=document.analysis,
                    download_url=document.download_url,
                )
                for document in documents
            ],
            final_summary=final_summary,
        )
        logger.info(
            "Dataset retrieve finished successfully: conversation_id=%s response_documents=%s",
            conversation_session.conversation_id,
            len(response.documents),
        )
        return DatasetRetrieveEnvelope(
            code="0",
            status=200,
            msg="",
            data=response.model_dump(by_alias=True),
            success=True,
        )
    except ValueError as exc:
        return _dataset_retrieve_error_response(400, str(exc))
    except IncompleteRead as exc:
        partial = exc.partial.decode("utf-8", errors="ignore") if exc.partial else ""
        return _dataset_retrieve_error_response(
            502,
            "Upstream response was interrupted before completion."
            + (f" Partial body: {partial}" if partial else ""),
        )
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        return _dataset_retrieve_error_response(
            exc.code,
            error_body or f"Dify retrieve request failed with status {exc.code}.",
        )
    except SQLAlchemyError as exc:
        return _dataset_retrieve_error_response(
            503,
            f"Unable to query document segments: {exc}",
        )
    except error.URLError as exc:
        return _dataset_retrieve_error_response(
            503,
            f"Unable to connect to Dify dataset service: {exc.reason}",
        )
    except ssl.SSLError as exc:
        return _dataset_retrieve_error_response(
            503,
            f"SSL error while connecting to Dify dataset service: {exc}",
        )
    except json.JSONDecodeError as exc:
        return _dataset_retrieve_error_response(
            502,
            f"Invalid JSON response from Dify dataset service: {exc}",
        )
    except Exception:
        logger.exception(
            "Unexpected error during dataset retrieve: dataset_id=%s conversation_id=%s",
            payload.dataset_id,
            payload.conversation_id or "",
        )
        return _dataset_retrieve_error_response(500, "Internal server error.")


@router.get("/conversations/{conversation_id}/summary", response_model=ConversationSummaryEnvelope)
def summarize_conversation(
    conversation_id: str,
    session: DbSession,
) -> ConversationSummaryEnvelope | JSONResponse:
    try:
        logger.info("Conversation summary started: conversation_id=%s", conversation_id)
        conversation_session = load_or_create_session(conversation_id)
        logger.info(
            "Conversation summary session loaded: conversation_id=%s turn_count=%s title_present=%s",
            conversation_session.conversation_id,
            len(conversation_session.turns),
            bool(conversation_session.title),
        )
        if not conversation_session.turns:
            return _dataset_retrieve_error_response(404, "Conversation has no turns.")

        summary_markdown = summarize_conversation_history(
            conversation_title=conversation_session.title,
            turns=[
                {
                    "turnId": turn.turn_id,
                    "query": turn.query,
                    "retrievalQuery": turn.retrieval_query,
                    "finalSummary": turn.final_summary,
                    "createdAt": turn.created_at.isoformat(),
                }
                for turn in conversation_session.turns
            ],
        )
        document_ids: list[UUID] = []
        seen_document_ids: set[UUID] = set()
        document_name_map: dict[UUID, str | None] = {}
        for turn in conversation_session.turns:
            for document in turn.document_metadata:
                try:
                    metadata_document_id = UUID(str(document.document_id))
                except ValueError:
                    logger.warning(
                        "Skipping invalid document metadata id in conversation summary: conversation_id=%s document_id=%s",
                        conversation_session.conversation_id,
                        document.document_id,
                    )
                    continue
                document_name_map.setdefault(metadata_document_id, document.document_name)
            for document_id_value in turn.document_ids:
                try:
                    document_id = UUID(str(document_id_value))
                except ValueError:
                    logger.warning(
                        "Skipping invalid document id in conversation summary: conversation_id=%s document_id=%s",
                        conversation_session.conversation_id,
                        document_id_value,
                    )
                    continue
                if document_id in seen_document_ids:
                    continue
                seen_document_ids.add(document_id)
                document_ids.append(document_id)
        logger.info(
            "Conversation summary collected document ids: conversation_id=%s unique_documents=%s",
            conversation_session.conversation_id,
            len(document_ids),
        )

        summary_documents: list[ConversationSummaryDocument] = []
        if document_ids:
            stmt = (
                select(DocumentSegment)
                .where(DocumentSegment.document_id.in_(document_ids))
                .order_by(
                    DocumentSegment.document_id.asc(),
                    DocumentSegment.position.asc(),
                )
            )
            segments = session.scalars(stmt).all()
            content_map: dict[UUID, list[str]] = {document_id: [] for document_id in document_ids}
            for segment in segments:
                content_map.setdefault(segment.document_id, []).append(segment.content)

            summary_documents = [
                ConversationSummaryDocument(
                    document_id=document_id,
                    document_name=document_name_map.get(document_id),
                    download_url=export_document_content(
                        document_id=str(document_id),
                        document_name=document_name_map.get(document_id) or str(document_id),
                        content="\n".join(content_map.get(document_id, [])),
                    )[1],
                )
                for document_id in document_ids
                if content_map.get(document_id)
            ]
        logger.info(
            "Conversation summary prepared document downloads: conversation_id=%s exported_documents=%s",
            conversation_session.conversation_id,
            len(summary_documents),
        )
        logger.info(
            "Conversation summary generated: conversation_id=%s summary_length=%s",
            conversation_session.conversation_id,
            len(summary_markdown),
        )

        response = ConversationSummaryResponse(
            conversation_id=conversation_session.conversation_id,
            conversation_title=conversation_session.title,
            summary_markdown=summary_markdown,
            documents=summary_documents,
        )
        logger.info(
            "Conversation summary finished successfully: conversation_id=%s",
            conversation_session.conversation_id,
        )
        return ConversationSummaryEnvelope(
            code="0",
            status=200,
            msg="",
            data=response.model_dump(by_alias=True),
            success=True,
        )
    except ValueError as exc:
        return _dataset_retrieve_error_response(400, str(exc))
    except Exception:
        logger.exception(
            "Unexpected error during conversation summary: conversation_id=%s",
            conversation_id,
        )
        return _dataset_retrieve_error_response(500, "Internal server error.")
