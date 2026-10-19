"""Messages endpoints for agent loops, tool calling, and canonical completions."""

from fastapi import APIRouter, Depends

from unified_llm.api.deps import get_request_context, get_service
from unified_llm.core.context import RequestContext
from unified_llm.core.errors import UnsupportedFeature
from unified_llm.core.schemas import MessageRequest, MessageResponse
from unified_llm.core.service import LLMService

router = APIRouter(prefix="/v1", tags=["messages"])


@router.post("/messages", response_model=MessageResponse)
async def create_message(
    request: MessageRequest,
    service: LLMService = Depends(get_service),
    ctx: RequestContext = Depends(get_request_context),
) -> MessageResponse:
    """Execute a canonical message completion for agents, tools, or dialogues."""
    return await service.execute(request, ctx)


@router.post("/messages/stream")
async def stream_message(
    request: MessageRequest,
    ctx: RequestContext = Depends(get_request_context),
) -> None:
    """Server-Sent Events streaming message completion (Phase 2 feature)."""
    raise UnsupportedFeature(
        "Streaming completion (/v1/messages/stream) is scheduled for Phase 2.",
        provider=request.provider,
    )



