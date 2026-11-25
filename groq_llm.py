"""Groq support.

CrewAI has no native Groq provider, so its OpenAI provider is pointed at Groq's
OpenAI-compatible endpoint, minus the request fields Groq does not document.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from crewai.llms.hooks.base import BaseInterceptor
from crewai.llms.providers.openai.completion import OpenAICompletion

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_MESSAGE_FIELDS = {"role", "content", "tool_calls", "tool_call_id"}


class AnswerSentAsToolCall(BaseInterceptor[httpx.Request, httpx.Response]):
    """gpt-oss sometimes sends its final answer as a call to a tool that does not exist.

    Groq rejects that with 400 tool_use_failed, but the error contains the complete
    answer, so it is turned back into an ordinary text reply. A broken call to a real
    tool becomes a 503, which the SDK retries.
    """

    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        return message

    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        if message.status_code != 400:
            return message
        message.read()
        return self._repair(message)

    async def aon_outbound(self, message: httpx.Request) -> httpx.Request:
        return message

    async def aon_inbound(self, message: httpx.Response) -> httpx.Response:
        if message.status_code != 400:
            return message
        await message.aread()
        return self._repair(message)

    @staticmethod
    def _repair(response: httpx.Response) -> httpx.Response:
        try:
            error = response.json()["error"]
            if error.get("code") != "tool_use_failed":
                return response
        except (ValueError, KeyError, TypeError, AttributeError):
            return response
        try:
            call = json.loads(error["failed_generation"])
            answer, name = call["arguments"], call["name"]
        except (ValueError, KeyError, TypeError):
            return httpx.Response(503, json={"error": error})
        # Only a call to an unknown tool is an answer in disguise.
        if "not in request.tools" not in str(error.get("message")):
            return httpx.Response(503, json={"error": error})
        content = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
        return httpx.Response(
            200,
            json={
                "id": f"repaired-{name}",
                "object": "chat.completion",
                "created": 0,
                "model": "",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )


class GroqCompletion(OpenAICompletion):
    def _prepare_completion_params(self, messages: Any, tools: Any = None) -> dict[str, Any]:
        params = super()._prepare_completion_params(messages, tools)
        # Groq answers 400 to messages[].name.
        params["messages"] = [
            {k: v for k, v in message.items() if k in _MESSAGE_FIELDS}
            for message in params["messages"]
        ]
        for tool in params.get("tools", []):
            tool["function"].pop("strict", None)
        return params


def build_groq_llm(
    model: str, api_key: str, max_tokens: int, base_url: str = GROQ_BASE_URL
) -> GroqCompletion:
    options: dict[str, Any] = {}
    if "gpt-oss" in model:  # reasoning tokens count against the per-minute limit
        options["additional_params"] = {"reasoning_effort": "low"}
    return GroqCompletion(
        model=model,
        api_key=api_key,
        base_url=base_url,
        custom_openai=True,
        max_completion_tokens=max_tokens,
        interceptor=AnswerSentAsToolCall(),
        max_retries=8,  # on 429 the SDK waits as long as Groq's retry-after header says
        **options,
    )
