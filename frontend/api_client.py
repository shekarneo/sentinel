"""Web console HTTP client for REST API access."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import FastAPI


class ApiClient:
    """Call versioned REST endpoints through ASGI transport."""

    def __init__(self, app: FastAPI, base_url: str = "http://console") -> None:
        self._transport = httpx.ASGITransport(app=app)
        self._base_url = base_url

    async def get_json(self, path: str) -> tuple[int, Any]:
        """Perform a GET request and return status code and JSON body."""
        async with httpx.AsyncClient(
            transport=self._transport,
            base_url=self._base_url,
        ) as client:
            response = await client.get(path)
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.status_code, response.json()
            return response.status_code, response.text
