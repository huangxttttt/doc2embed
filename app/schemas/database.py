from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class TableInfo(BaseModel):
    schema_name: str = Field(alias="schema", serialization_alias="schema")
    name: str
    columns: list[str]

    model_config = ConfigDict(populate_by_name=True)


class TableRowsResponse(BaseModel):
    table: str
    schema_name: str = Field(alias="schema", serialization_alias="schema")
    limit: int = Field(ge=1, le=100)
    rows: list[dict[str, object]]

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)


class DocumentSegmentRead(BaseModel):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    document_id: UUID
    position: int
    content: str
    word_count: int
    tokens: int
    keywords: dict | list | None
    index_node_id: str | None
    index_node_hash: str | None
    hit_count: int
    enabled: bool
    disabled_at: datetime | None
    disabled_by: UUID | None
    status: str
    created_by: UUID
    created_at: datetime
    indexing_at: datetime | None
    completed_at: datetime | None
    error: str | None
    stopped_at: datetime | None
    answer: str | None
    updated_by: UUID | None
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DatasetRetrieveRequest(BaseModel):
    dataset_id: str = Field(alias="datasetId", serialization_alias="datasetId")
    query: str = Field(min_length=1)
    conversation_id: str | None = Field(
        default=None,
        alias="conversationId",
        serialization_alias="conversationId",
    )

    model_config = ConfigDict(populate_by_name=True)


class RetrievedDocumentAnswer(BaseModel):
    analysis: str | None = None
    download_url: str | None = Field(
        default=None,
        alias="downloadUrl",
        serialization_alias="downloadUrl",
    )

    model_config = ConfigDict(populate_by_name=True)


class RetrievedDocumentContent(BaseModel):
    document_id: UUID = Field(alias="documentId", serialization_alias="documentId")
    document_name: str | None = Field(
        default=None,
        alias="documentName",
        serialization_alias="documentName",
    )
    content: str
    analysis: str | None = None
    storage_path: str | None = Field(
        default=None,
        alias="storagePath",
        serialization_alias="storagePath",
    )
    download_url: str | None = Field(
        default=None,
        alias="downloadUrl",
        serialization_alias="downloadUrl",
    )

    model_config = ConfigDict(populate_by_name=True)


class DatasetRetrieveResponse(BaseModel):
    query: str
    conversation_id: str = Field(
        alias="conversationId",
        serialization_alias="conversationId",
    )
    conversation_title: str = Field(
        default="",
        alias="conversationTitle",
        serialization_alias="conversationTitle",
    )
    documents: list[RetrievedDocumentAnswer]
    final_summary: str = Field(
        alias="finalSummary",
        serialization_alias="finalSummary",
    )

    model_config = ConfigDict(populate_by_name=True)


class DatasetRetrieveEnvelope(BaseModel):
    code: str
    status: int
    msg: str
    data: DatasetRetrieveResponse | dict[str, object]
    success: bool


class ConversationSummaryDocument(BaseModel):
    document_id: UUID = Field(alias="documentId", serialization_alias="documentId")
    document_name: str | None = Field(
        default=None,
        alias="documentName",
        serialization_alias="documentName",
    )
    download_url: str = Field(alias="downloadUrl", serialization_alias="downloadUrl")

    model_config = ConfigDict(populate_by_name=True)


class ConversationTurnDocument(BaseModel):
    document_id: str = Field(alias="documentId", serialization_alias="documentId")
    document_name: str | None = Field(
        default=None,
        alias="documentName",
        serialization_alias="documentName",
    )

    model_config = ConfigDict(populate_by_name=True)


class ConversationSummaryResponse(BaseModel):
    conversation_id: str = Field(
        alias="conversationId",
        serialization_alias="conversationId",
    )
    conversation_title: str = Field(
        default="",
        alias="conversationTitle",
        serialization_alias="conversationTitle",
    )
    summary_markdown: str = Field(
        alias="summaryMarkdown",
        serialization_alias="summaryMarkdown",
    )
    documents: list[ConversationSummaryDocument] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class ConversationSummaryEnvelope(BaseModel):
    code: str
    status: int
    msg: str
    data: ConversationSummaryResponse | dict[str, object]
    success: bool


class ConversationTurn(BaseModel):
    document_metadata: list[ConversationTurnDocument] = Field(
        default_factory=list,
        alias="documentMetadata",
        serialization_alias="documentMetadata",
    )
    turn_id: str = Field(alias="turnId", serialization_alias="turnId")
    query: str
    retrieval_query: str = Field(
        alias="retrievalQuery",
        serialization_alias="retrievalQuery",
    )
    final_summary: str = Field(
        alias="finalSummary",
        serialization_alias="finalSummary",
    )
    created_at: datetime = Field(alias="createdAt", serialization_alias="createdAt")
    document_ids: list[str] = Field(
        default_factory=list,
        alias="documentIds",
        serialization_alias="documentIds",
    )

    model_config = ConfigDict(populate_by_name=True)


class ConversationSession(BaseModel):
    conversation_id: str = Field(
        alias="conversationId",
        serialization_alias="conversationId",
    )
    title: str = ""
    memory_summary: str = Field(
        default="",
        alias="memorySummary",
        serialization_alias="memorySummary",
    )
    turns: list[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(alias="createdAt", serialization_alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt", serialization_alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True)
