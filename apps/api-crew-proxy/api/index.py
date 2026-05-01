"""Thin reverse proxy on Vercel: forwards to ``CREW_BACKEND_URL`` (full Crew worker).

Use when ``apps/api-crew`` exceeds the 250 MiB serverless cap: deploy the heavy
stack on Railway / Fly / Cloud Run, set ``CREW_BACKEND_URL`` here, and point
``NEXT_PUBLIC_CREW_API_URL`` at this project's URL.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }
)


def _backend_base() -> str:
    return (os.environ.get("CREW_BACKEND_URL") or "").strip().rstrip("/")


def _forward_headers(request: Request) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in request.headers.items():
        lk = key.lower()
        if lk in _HOP_BY_HOP or lk == "host":
            continue
        out[key] = value
    return out


@asynccontextmanager
async def _lifespan(app: Starlette) -> AsyncIterator[None]:
    timeout = httpx.Timeout(300.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        app.state.http = client
        yield


async def _proxy(request: Request) -> Response:
    base = _backend_base()
    if not base:
        return Response(
            b'{"detail":"CREW_BACKEND_URL is not set"}',
            status_code=503,
            media_type="application/json",
        )
    client: httpx.AsyncClient = request.app.state.http
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"
    url = f"{base}{path}"
    hdrs = _forward_headers(request)

    async def body_stream() -> AsyncIterator[bytes]:
        async for chunk in request.stream():
            yield chunk

    try:
        req = client.build_request(
            request.method,
            url,
            headers=hdrs,
            content=body_stream(),
        )
        resp = await client.send(req, stream=True)
    except httpx.RequestError as e:
        return Response(
            f'{{"detail":"upstream error: {e!s}"}}'.encode(),
            status_code=502,
            media_type="application/json",
        )

    passthrough: Mapping[str, str] = {
        k: v
        for k, v in resp.headers.items()
        if k.lower() not in _HOP_BY_HOP
    }

    async def stream_body() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_raw():
                yield chunk
        finally:
            await resp.aclose()

    return StreamingResponse(
        stream_body(),
        status_code=resp.status_code,
        headers=dict(passthrough),
    )


app = Starlette(
    routes=[Route("/{path:path}", endpoint=_proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])],
    lifespan=_lifespan,
)
