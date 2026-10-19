"""Chat completion endpoints."""

from fastapi import APIRouter, Depends

from unified_llm.api.deps import get_request_context, get_service
from unified_llm.core.context import RequestContext
from unified_llm.core.errors import UnsupportedFeature
from unified_llm.core.schemas import ChatRequest, ChatResponse
from unified_llm.core.service import LLMService

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    service: LLMService = Depends(get_service),
    ctx: RequestContext = Depends(get_request_context),
) -> ChatResponse:
    """Execute a canonical non-streaming chat completion."""
    return await service.chat(request, ctx)


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    ctx: RequestContext = Depends(get_request_context),
) -> None:
    """Server-Sent Events streaming chat completion (Phase 2 feature)."""
    raise UnsupportedFeature(
        "Streaming completion (/v1/chat/stream) is scheduled for Phase 2.",
        provider=request.provider,
    )
