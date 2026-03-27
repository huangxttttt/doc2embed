from http.client import IncompleteRead
import json
from typing import Any
from urllib import error, request

from fastapi import HTTPException

from app.core.config import settings


def _normalize_text(value: str) -> str:
    return " ".join((value or "").strip().split())


def _limit_retrieval_query_length(query: str, max_length: int = 250) -> str:
    normalized = _normalize_text(query)
    if len(normalized) <= max_length:
        return normalized
    return normalized[:max_length].rstrip()


def _parse_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _post_chat_completion(messages: list[dict[str, Any]]) -> dict[str, Any]:
    if not settings.llm_api_key:
        raise HTTPException(status_code=500, detail="LLM API key is not configured.")

    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = json.dumps(
        {
            "model": settings.llm_model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
    ).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.llm_api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=settings.llm_timeout_seconds) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
    except IncompleteRead as exc:
        partial = exc.partial.decode("utf-8", errors="ignore") if exc.partial else ""
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM response was interrupted before completion."
                + (f" Partial body: {partial}" if partial else "")
            ),
        ) from exc
    except error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore")
        raise HTTPException(
            status_code=exc.code,
            detail=error_body or f"LLM request failed with status {exc.code}.",
        ) from exc
    except error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unable to connect to LLM service: {exc.reason}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid JSON response from LLM service: {exc}",
        ) from exc

    choices = data.get("choices") or []
    if not choices:
        raise HTTPException(status_code=502, detail="LLM response did not contain choices.")

    message = choices[0].get("message") or {}
    content = _parse_message_content(message.get("content"))
    if not content:
        raise HTTPException(status_code=502, detail="LLM response content was empty.")

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"LLM returned non-JSON content: {exc}",
        ) from exc


def _chunk_documents(
    documents: list[dict[str, Any]],
    batch_size: int,
) -> list[list[dict[str, Any]]]:
    return [
        documents[index:index + batch_size]
        for index in range(0, len(documents), batch_size)
    ]


def _normalize_conversation_context(
    conversation_context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not conversation_context:
        return {
            "title": "",
            "memorySummary": "",
            "recentTurns": [],
            "turnCount": 0,
        }
    return {
        "title": str(conversation_context.get("title", "")),
        "memorySummary": str(conversation_context.get("memorySummary", "")),
        "recentTurns": list(conversation_context.get("recentTurns", [])),
        "turnCount": int(conversation_context.get("turnCount", 0)),
    }


def _fallback_title(query: str) -> str:
    normalized = " ".join(query.strip().split())
    if not normalized:
        return "新会话"
    return normalized[:20]


def generate_conversation_title(query: str) -> str:
    try:
        response = _post_chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你需要根据用户的首轮提问生成一个简短的中文会话标题。"
                        "标题要准确概括问题主题，适合作为聊天列表标题。"
                        "长度尽量控制在12个汉字以内，不要使用书名号、引号、句号，不要输出解释。"
                        "只返回 JSON，对象顶层键为 title。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "output_schema": {"title": "string"},
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        title = str(response.get("title", "")).strip()
        return title or _fallback_title(query)
    except HTTPException:
        return _fallback_title(query)


def expand_query_for_retrieval(
    query: str,
    conversation_context: dict[str, Any] | None = None,
) -> str:
    normalized_query = _limit_retrieval_query_length(query)
    normalized_context = _normalize_conversation_context(conversation_context)
    if normalized_context["turnCount"] <= 0 and not normalized_context["memorySummary"]:
        return normalized_query

    try:
        response = _post_chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你需要根据多轮对话背景，对用户当前 query 做检索用的上下文补全。"
                        "你的目标是让检索 query 更完整、更明确、更容易命中相关内容。"
                        "只能补全 conversationContext 中已经明确出现过的背景、指代关系、主题和实体。"
                        "不允许新增用户未提及的新主题，不允许改变原始问题意图，不允许补充外部知识。"
                        "输出应是一条适合向量检索的单句 query，不要输出解释。"
                        "只返回 JSON，对象顶层键为 expandedQuery。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "conversationContext": normalized_context,
                            "output_schema": {"expandedQuery": "string"},
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        expanded_query = _limit_retrieval_query_length(
            str(response.get("expandedQuery", ""))
        )
        return expanded_query or normalized_query
    except HTTPException:
        return normalized_query


def analyze_documents(
    query: str,
    documents: list[dict[str, Any]],
    conversation_context: dict[str, Any] | None = None,
) -> tuple[dict[str, str], str]:
    normalized_context = _normalize_conversation_context(conversation_context)
    if not documents:
        if normalized_context["turnCount"] <= 0 and not normalized_context["memorySummary"]:
            return {}, ""

        summary_response = _post_chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你需要在没有新文档分析结果的情况下，基于已有多轮 conversationContext，"
                        "围绕用户当前问题输出一份连续、详细的 Markdown 回复。"
                        "只能依赖 conversationContext 中已有的历史总结内容，不允许补充外部知识或编造事实。"
                        "不要提及来源、轮次或材料编号，直接回答用户问题。"
                        "只返回 JSON，对象顶层键为 summary。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "conversationContext": normalized_context,
                            "output_schema": {"summary": "string"},
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        return {}, str(summary_response.get("summary", "")).strip()

    analysis_by_id: dict[str, str] = {}
    truncated_documents = [
        {
            "documentId": document["documentId"],
            "documentName": document.get("documentName"),
            "content": document["content"][: settings.llm_max_document_chars],
        }
        for document in documents
    ]

    for batch in _chunk_documents(truncated_documents, max(settings.llm_batch_size, 1)):
        response = _post_chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "你需要结合用户当前问题，对单篇文档进行详细分析。"
                        "如果提供了 conversationContext，它只能帮助你理解多轮对话背景和用户意图延续，不能替代当前文档内容成为事实依据。"
                        "回答必须严格以当前这篇文档的全文内容为背景和依据，不允许跳出文档范围。"
                        "你只能依据提供的文档内容生成，不允许补充外部知识、背景信息、常识推断、猜测或编造内容。"
                        "分析必须围绕当前 query 展开，说明这篇文档与 query 的关系、能直接支撑的事实和可得出的结论。"
                        "分析内容要尽量详细、具体、完整，但所有表述都必须能在文档内容中找到依据。"
                        "只返回 JSON，对象顶层键为 documents。"
                        "每个元素必须包含 documentId 和 analysis。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "conversationContext": normalized_context,
                            "documents": batch,
                            "output_schema": {
                                "documents": [
                                    {
                                        "documentId": "string",
                                        "analysis": "string",
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        for item in response.get("documents", []):
            document_id = item.get("documentId")
            analysis = item.get("analysis")
            if document_id and analysis:
                analysis_by_id[str(document_id)] = str(analysis)

    summary_response = _post_chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你需要基于多篇文档 analysis 的结果，对用户当前问题做最终详细回复，并输出 Markdown。"
                    "如果提供了 conversationContext，它只用于维持多轮对话连续性和承接历史 finalSummary，不是独立事实来源。"
                    "最终回复必须以 documentAnalyses 为主背景，并结合当前 query 直接组织答案。"
                    "可以吸收 conversationContext 中的历史 finalSummary 语义，但不能脱离 documentAnalyses 和当前 query 单独发挥。"
                    "不要提及信息来自哪一篇文档、哪一条 analysis、哪一轮历史总结，也不要写来源说明。"
                    "不要输出“哪些内容无法确认”这类单独的保留说明。"
                    "内容要详细、连贯、完整，直接围绕用户问题作答。"
                    "不允许补充任何外部事实、背景知识、主观假设、常识延伸或编造结论。"
                    "只返回 JSON，对象顶层键为 summary。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "conversationContext": normalized_context,
                        "documentAnalyses": [
                            {
                                "documentId": document["documentId"],
                                "documentName": document.get("documentName"),
                                "analysis": analysis_by_id.get(str(document["documentId"]), ""),
                            }
                            for document in truncated_documents
                        ],
                        "output_schema": {"summary": "string"},
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    )
    summary = str(summary_response.get("summary", "")).strip()
    return analysis_by_id, summary


def summarize_conversation_memory(
    query: str,
    existing_memory: str,
    recent_turns: list[dict[str, Any]],
    latest_final_summary: str,
) -> str:
    response = _post_chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你需要为多轮对话生成一份滚动会话记忆摘要。"
                    "目标是尽可能保留历轮 finalSummary 的关键信息，同时压缩成后续轮次可复用的短摘要。"
                    "只保留对后续回答仍然有价值的核心结论、用户关注点、持续约束和上下文延续。"
                    "不要保留措辞修饰、重复表达、来源说明或文档编号。"
                    "不要补充外部知识，不要编造内容。"
                    "返回的 memorySummary 应为一段紧凑但信息密度高的中文文本。"
                    "只返回 JSON，对象顶层键为 memorySummary。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "query": query,
                        "existingMemory": existing_memory,
                        "recentTurns": recent_turns,
                        "latestFinalSummary": latest_final_summary,
                        "output_schema": {"memorySummary": "string"},
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    )
    return str(response.get("memorySummary", "")).strip()


def summarize_conversation_history(
    conversation_title: str,
    turns: list[dict[str, Any]],
) -> str:
    if not turns:
        return (
            "## 基本情况\n\n暂无可用会话内容。\n\n"
            "## 事件概括\n\n暂无可用会话内容。\n\n"
            "## 情报价值评估\n\n暂无可用会话内容。\n\n"
            "## 原文内容\n\n暂无可用会话内容。\n\n"
            "## 情报摘要\n\n暂无可用会话内容。\n\n"
            "## 情报类别\n\n暂无可用会话内容。"
        )

    response = _post_chat_completion(
        [
            {
                "role": "system",
                "content": (
                    "你需要基于同一会话的多轮 finalSummary，输出一份情报型 Markdown 总结。"
                    "只能依赖提供的会话标题和各轮 finalSummary，不允许补充外部知识、背景信息、猜测或编造内容。"
                    "输出必须严格包含以下六个一级标题，且顺序不能变："
                    "基本情况、事件概括、情报价值评估、原文内容、情报摘要、情报类别。"
                    "“原文内容”部分应整理呈现多轮 finalSummary 中的重要原始表述和核心内容，不要写成空泛概括。"
                    "“情报类别”部分给出最合适的类别归纳，可为一个或多个类别。"
                    "全文使用 Markdown 输出，不要添加这六个标题以外的一级标题，不要写来源说明。"
                    "只返回 JSON，对象顶层键为 summaryMarkdown。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "conversationTitle": conversation_title,
                        "turns": turns,
                        "output_schema": {"summaryMarkdown": "string"},
                    },
                    ensure_ascii=False,
                ),
            },
        ]
    )
    summary_markdown = str(response.get("summaryMarkdown", "")).strip()
    return summary_markdown
