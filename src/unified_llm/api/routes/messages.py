"""Messages endpoints for agent loops, tool calling, and canonical completions."""

import json
from typing import AsyncIterator
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from unified_llm.api.deps import get_request_context, get_service
from unified_llm.core.context import RequestContext
from unified_llm.core.errors import ULILError
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
    service: LLMService = Depends(get_service),
    ctx: RequestContext = Depends(get_request_context),
) -> StreamingResponse:
    """Execute a Server-Sent Events (SSE) streaming message completion."""

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in service.stream(request, ctx):
                yield f"event: {event.event}\ndata: {json.dumps(event.data)}\n\n"
        except ULILError as exc:
            yield f"event: error\ndata: {json.dumps(exc.to_envelope(request_id=ctx.request_id))}\n\n"
        except Exception as exc:
            err_envelope = {
                "error": {
                    "type": "internal_error",
                    "message": str(exc),
                    "provider": request.provider,
                    "retryable": False,
                    "request_id": ctx.request_id,
                }
            }
            yield f"event: error\ndata: {json.dumps(err_envelope)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )




