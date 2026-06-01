"""Tests for OpenAPI/Swagger toolset auto-generation.

A small OpenAPI 3.0 spec with two operations exercises the schema generation
(path + query + body params with correct required flags), request building
(URL substitution, query params, JSON body), error handling (non-2xx returns an
``error:`` string rather than raising), YAML parsing, and the ``operations=``
allowlist filter. Responses are faked with ``httpx.MockTransport`` so no network
is touched.
"""

from __future__ import annotations

import json

import httpx
import pytest

from yaab.tools.openapi import OpenAPITool, openapi_toolset
from yaab.types import RunContext

SPEC: dict = {
    "openapi": "3.0.0",
    "info": {"title": "Pet Store", "version": "1.0.0"},
    "servers": [{"url": "https://api.petstore.test/v1"}],
    "paths": {
        "/pets/{petId}": {
            "get": {
                "operationId": "getPet",
                "summary": "Get a pet by id",
                "description": "Returns a single pet.",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                        "description": "id of the pet",
                    },
                    {
                        "name": "verbose",
                        "in": "query",
                        "required": False,
                        "schema": {"type": "boolean"},
                        "description": "include extra detail",
                    },
                ],
            }
        },
        "/pets": {
            "post": {
                "operationId": "createPet",
                "summary": "Create a pet",
                "description": "Adds a new pet to the store.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "tag": {"type": "string"},
                                },
                            }
                        }
                    },
                },
            }
        },
    },
}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _by_name(tools) -> dict[str, OpenAPITool]:
    return {t.name: t for t in tools}


def test_toolset_builds_one_tool_per_operation():
    tools = openapi_toolset(SPEC)
    names = _by_name(tools)
    assert set(names) == {"getPet", "createPet"}
    assert all(isinstance(t, OpenAPITool) for t in tools)


def test_description_combines_summary_and_description():
    tools = _by_name(openapi_toolset(SPEC))
    desc = tools["getPet"].description
    assert "Get a pet by id" in desc
    assert "Returns a single pet." in desc


def test_schema_has_path_and_query_params_with_required_flags():
    tools = _by_name(openapi_toolset(SPEC))
    schema = tools["getPet"].schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "getPet"
    props = fn["parameters"]["properties"]
    assert set(props) == {"petId", "verbose"}
    # path param is required, optional query param is not.
    assert fn["parameters"]["required"] == ["petId"]
    assert props["petId"]["type"] == "integer"
    assert props["verbose"]["type"] == "boolean"


def test_schema_merges_request_body_properties():
    tools = _by_name(openapi_toolset(SPEC))
    schema = tools["createPet"].schema()["function"]
    props = schema["parameters"]["properties"]
    assert set(props) == {"name", "tag"}
    assert schema["parameters"]["required"] == ["name"]


@pytest.mark.asyncio
async def test_execute_get_substitutes_path_and_sends_query():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        return httpx.Response(200, json={"id": 42, "name": "Rex"})

    tools = _by_name(openapi_toolset(SPEC, client=_client(handler)))
    result = await tools["getPet"].execute(RunContext(), petId=42, verbose=True)

    assert result == {"id": 42, "name": "Rex"}
    assert captured["method"] == "GET"
    assert captured["url"].startswith("https://api.petstore.test/v1/pets/42")
    assert "verbose=true" in captured["url"]


@pytest.mark.asyncio
async def test_execute_post_sends_json_body():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content)
        return httpx.Response(201, json={"id": 7, "name": "Milo"})

    tools = _by_name(openapi_toolset(SPEC, client=_client(handler)))
    result = await tools["createPet"].execute(RunContext(), name="Milo", tag="dog")

    assert result == {"id": 7, "name": "Milo"}
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.petstore.test/v1/pets"
    assert captured["body"] == {"name": "Milo", "tag": "dog"}


@pytest.mark.asyncio
async def test_static_headers_are_sent():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={})

    tools = _by_name(
        openapi_toolset(SPEC, client=_client(handler), headers={"X-API-Key": "secret"})
    )
    await tools["getPet"].execute(RunContext(), petId=1)
    assert captured["auth"] == "secret"


@pytest.mark.asyncio
async def test_non_2xx_returns_error_string_not_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="pet not found")

    tools = _by_name(openapi_toolset(SPEC, client=_client(handler)))
    result = await tools["getPet"].execute(RunContext(), petId=999)
    assert isinstance(result, str)
    assert result.startswith("error:")
    assert "404" in result
    assert "pet not found" in result


@pytest.mark.asyncio
async def test_text_response_returned_when_not_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="plain text body")

    tools = _by_name(openapi_toolset(SPEC, client=_client(handler)))
    result = await tools["getPet"].execute(RunContext(), petId=1)
    assert result == "plain text body"


@pytest.mark.asyncio
async def test_base_url_override_takes_precedence():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(200, json={})

    tools = _by_name(
        openapi_toolset(SPEC, base_url="https://override.test/api", client=_client(handler))
    )
    await tools["getPet"].execute(RunContext(), petId=5)
    assert captured["url"].startswith("https://override.test/api/pets/5")


def test_yaml_string_spec_parses():
    yaml_spec = """
openapi: "3.0.0"
info:
  title: Tiny
  version: "1.0.0"
servers:
  - url: https://yaml.test
paths:
  /ping:
    get:
      operationId: ping
      summary: ping it
"""
    tools = _by_name(openapi_toolset(yaml_spec))
    assert "ping" in tools
    assert "ping it" in tools["ping"].description


def test_json_string_spec_parses():
    tools = _by_name(openapi_toolset(json.dumps(SPEC)))
    assert set(tools) == {"getPet", "createPet"}


def test_operations_allowlist_filters():
    tools = _by_name(openapi_toolset(SPEC, operations=["getPet"]))
    assert set(tools) == {"getPet"}


def test_operation_id_fallback_slug():
    spec = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://x.test"}],
        "paths": {
            "/things/{id}": {
                "get": {"summary": "no operation id here"},
            }
        },
    }
    tools = openapi_toolset(spec)
    assert len(tools) == 1
    # Fallback name is a method_path slug, no braces/slashes left.
    name = tools[0].name
    assert "get" in name.lower()
    assert "/" not in name and "{" not in name and "}" not in name
