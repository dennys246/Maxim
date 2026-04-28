"""Tests for CC9 — Tool input_schema dual-format support.

Covers:

* Custom format (legacy) is preserved verbatim and dispatches as before.
* JSONSchema input is auto-detected, normalized to custom format
  internally, and the original is preserved for export.
* ``Tool.to_json_schema()`` exports a valid, MCP-compatible JSONSchema
  dict from either format of authored schema.
* Round-trip stability: ``custom → JSONSchema → custom`` preserves the
  dispatch semantics (required vs optional, type used for dispatch).
"""

from __future__ import annotations

from typing import Any

import pytest

from maxim.tools.base import (
    Tool,
    ToolOutput,
    _custom_to_json_schema,
    _json_schema_to_custom,
    _looks_like_json_schema,
)


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures — tools authored in each format
# ─────────────────────────────────────────────────────────────────────────────


class _CustomFormatTool(Tool):
    """Authored with the legacy custom format."""

    name = "custom_fmt"
    description = "Test tool — custom format"
    input_schema: dict[str, Any] = {
        "path": str,
        "tail_lines": (int, None),
        "overwrite": (bool, False),
    }

    def execute(self, **kwargs: Any) -> Any:
        return {"path": kwargs.get("path"), "tail": kwargs.get("tail_lines")}


class _JsonSchemaTool(Tool):
    """Authored with JSONSchema."""

    name = "json_schema_fmt"
    description = "Test tool — JSONSchema format"
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "tail_lines": {"type": "integer", "default": None},
            "overwrite": {"type": "boolean", "default": False},
        },
        "required": ["path"],
    }

    def execute(self, **kwargs: Any) -> Any:
        return {"path": kwargs.get("path"), "tail": kwargs.get("tail_lines")}


class _NoSchemaTool(Tool):
    """Tool with no input_schema declared."""

    name = "no_schema"
    description = "no params"

    def execute(self, **kwargs: Any) -> Any:
        return "ok"


class _DescriptionValueTool(Tool):
    """Custom format using description-as-value pattern."""

    name = "desc_value"
    description = "uses string-as-description spec"
    input_schema: dict[str, Any] = {
        "context": "Optional description of what you're looking for",
    }

    def execute(self, **kwargs: Any) -> Any:
        return kwargs.get("context", "")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers under test
# ─────────────────────────────────────────────────────────────────────────────


class TestLooksLikeJsonSchema:
    def test_detects_jsonschema(self):
        assert _looks_like_json_schema({"type": "object", "properties": {"a": {"type": "string"}}})

    def test_detects_jsonschema_with_required(self):
        assert _looks_like_json_schema(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            }
        )

    def test_rejects_custom_format_with_type_param(self):
        # A user tool whose param happens to be named "type" — must NOT
        # be misclassified as JSONSchema (no "properties" key).
        assert not _looks_like_json_schema({"type": str, "name": str})

    def test_rejects_custom_format(self):
        assert not _looks_like_json_schema({"path": str, "limit": (int, 10)})

    def test_rejects_empty_dict(self):
        assert not _looks_like_json_schema({})

    def test_rejects_non_dict(self):
        assert not _looks_like_json_schema(None)
        assert not _looks_like_json_schema("not a dict")
        assert not _looks_like_json_schema([])

    def test_requires_object_type_not_array(self):
        # An array-typed JSONSchema isn't a top-level tool input schema.
        assert not _looks_like_json_schema({"type": "array", "properties": {"a": {"type": "string"}}})

    def test_requires_properties_key(self):
        # Just "type": "object" without "properties" isn't enough — could
        # be a malformed schema or a custom format with a literal "type" key.
        assert not _looks_like_json_schema({"type": "object"})


class TestJsonSchemaToCustom:
    def test_required_typed_param(self):
        custom = _json_schema_to_custom(
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            }
        )
        assert custom == {"path": str}

    def test_optional_with_default(self):
        custom = _json_schema_to_custom(
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["path"],
            }
        )
        assert custom == {"path": str, "limit": (int, 10)}

    def test_optional_without_default(self):
        custom = _json_schema_to_custom(
            {
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            }
        )
        assert custom == {"limit": (int, None)}

    def test_all_basic_types(self):
        custom = _json_schema_to_custom(
            {
                "type": "object",
                "properties": {
                    "s": {"type": "string"},
                    "i": {"type": "integer"},
                    "n": {"type": "number"},
                    "b": {"type": "boolean"},
                    "a": {"type": "array"},
                    "o": {"type": "object"},
                },
                "required": ["s", "i", "n", "b", "a", "o"],
            }
        )
        assert custom == {"s": str, "i": int, "n": float, "b": bool, "a": list, "o": dict}

    def test_type_union_collapses_to_first_non_null(self):
        # ["string", "null"] is common in OpenAI-style schemas. Collapse
        # to str for dispatch (validation only checks presence).
        custom = _json_schema_to_custom(
            {
                "type": "object",
                "properties": {"x": {"type": ["string", "null"]}},
                "required": ["x"],
            }
        )
        assert custom == {"x": str}

    def test_unknown_type_falls_back_to_dict(self):
        custom = _json_schema_to_custom(
            {
                "type": "object",
                "properties": {"x": {"type": "weirdcustom"}},
                "required": ["x"],
            }
        )
        assert custom == {"x": dict}

    def test_empty_properties(self):
        assert _json_schema_to_custom({"type": "object", "properties": {}}) == {}


class TestCustomToJsonSchema:
    def test_required_typed_param(self):
        out = _custom_to_json_schema({"path": str})
        assert out == {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    def test_optional_with_default(self):
        out = _custom_to_json_schema({"limit": (int, 10)})
        assert out == {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
        }

    def test_optional_with_none_default_omits_default_key(self):
        # Don't emit "default": null — noisy and conventionally optional
        # parameters omit the default entirely.
        out = _custom_to_json_schema({"limit": (int, None)})
        assert out == {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        }
        assert "required" not in out

    def test_description_as_value_pattern(self):
        out = _custom_to_json_schema({"context": "Optional description of what you're looking for"})
        assert out == {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "Optional description of what you're looking for",
                }
            },
            "required": ["context"],
        }

    def test_empty_schema(self):
        assert _custom_to_json_schema({}) == {"type": "object", "properties": {}}

    def test_required_omitted_when_empty(self):
        # JSONSchema convention: required is optional and should be omitted
        # when no params are required.
        out = _custom_to_json_schema({"x": (str, "default")})
        assert "required" not in out

    def test_required_listed_when_present(self):
        out = _custom_to_json_schema({"a": str, "b": (int, 0), "c": str})
        assert set(out["required"]) == {"a", "c"}
        assert "b" not in out["required"]


# ─────────────────────────────────────────────────────────────────────────────
# Tool ABC integration
# ─────────────────────────────────────────────────────────────────────────────


class TestToolConstructionPreservesAuthoredSchema:
    """input_schema is left exactly as the tool author declared it.

    The ``@maxim.tool`` decorator authors JSONSchema directly and downstream
    consumers (test_api_verbs.py, prompt builders, MCP exporters) read
    ``tool.input_schema`` expecting it back unchanged. Mutating it on
    construction would silently break that public-facing contract.
    """

    def test_custom_format_preserved(self):
        t = _CustomFormatTool()
        assert t.input_schema == {
            "path": str,
            "tail_lines": (int, None),
            "overwrite": (bool, False),
        }

    def test_jsonschema_preserved_verbatim(self):
        t = _JsonSchemaTool()
        assert t.input_schema == {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "tail_lines": {"type": "integer", "default": None},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["path"],
        }

    def test_no_schema_construction(self):
        t = _NoSchemaTool()
        assert t.input_schema == {}

    def test_construction_does_not_mutate_class_attribute(self):
        before = dict(_JsonSchemaTool.input_schema)
        _JsonSchemaTool()
        _JsonSchemaTool()
        assert _JsonSchemaTool.input_schema == before


class TestToolDispatchEquivalence:
    """JSONSchema input → tool executes with same dispatch behavior as
    the custom-format equivalent."""

    def test_custom_dispatch_required_missing(self):
        t = _CustomFormatTool()
        result = t.run()
        assert result.success is False
        assert "Missing required input: path" in (result.error or "")

    def test_jsonschema_dispatch_required_missing(self):
        t = _JsonSchemaTool()
        result = t.run()
        assert result.success is False
        assert "Missing required input: path" in (result.error or "")

    def test_custom_dispatch_optional_omitted_ok(self):
        t = _CustomFormatTool()
        result = t.run(path="/tmp/x")
        assert result.success is True
        assert result.output == {"path": "/tmp/x", "tail": None}

    def test_jsonschema_dispatch_optional_omitted_ok(self):
        t = _JsonSchemaTool()
        result = t.run(path="/tmp/x")
        assert result.success is True
        assert result.output == {"path": "/tmp/x", "tail": None}

    def test_custom_dispatch_all_params(self):
        t = _CustomFormatTool()
        result = t.run(path="/tmp/x", tail_lines=5, overwrite=True)
        assert result.success is True
        assert result.output == {"path": "/tmp/x", "tail": 5}

    def test_jsonschema_dispatch_all_params(self):
        t = _JsonSchemaTool()
        result = t.run(path="/tmp/x", tail_lines=5, overwrite=True)
        assert result.success is True
        assert result.output == {"path": "/tmp/x", "tail": 5}


class TestToJsonSchemaExport:
    def test_jsonschema_authored_returns_original(self):
        # When authored as JSONSchema, export returns the original
        # verbatim (preserves descriptions, defaults, etc.).
        t = _JsonSchemaTool()
        exported = t.to_json_schema()
        assert exported == _JsonSchemaTool.input_schema

    def test_jsonschema_export_is_deep_copy(self):
        # Caller mutations on the exported dict must not affect the
        # tool's input_schema or future exports.
        t = _JsonSchemaTool()
        exported = t.to_json_schema()
        exported["properties"]["path"]["type"] = "MUTATED"
        again = t.to_json_schema()
        assert again["properties"]["path"]["type"] == "string"
        assert t.input_schema["properties"]["path"]["type"] == "string"

    def test_custom_authored_exports_jsonschema(self):
        t = _CustomFormatTool()
        exported = t.to_json_schema()
        assert exported["type"] == "object"
        assert "properties" in exported
        assert exported["properties"]["path"] == {"type": "string"}
        assert exported["properties"]["tail_lines"] == {"type": "integer"}
        assert exported["properties"]["overwrite"] == {
            "type": "boolean",
            "default": False,
        }
        assert exported["required"] == ["path"]

    def test_no_schema_exports_empty_object(self):
        t = _NoSchemaTool()
        exported = t.to_json_schema()
        assert exported == {"type": "object", "properties": {}}

    def test_description_value_pattern_export(self):
        t = _DescriptionValueTool()
        exported = t.to_json_schema()
        # Description-as-value: required string with description attached.
        assert exported["properties"]["context"]["type"] == "string"
        assert exported["properties"]["context"]["description"] == "Optional description of what you're looking for"
        assert exported["required"] == ["context"]

    def test_export_is_mcp_compatible_shape(self):
        # MCP tool inputSchema spec: top-level "type": "object", "properties"
        # dict, "required" array (optional).
        for cls in (_CustomFormatTool, _JsonSchemaTool, _NoSchemaTool, _DescriptionValueTool):
            schema = cls().to_json_schema()
            assert schema["type"] == "object"
            assert isinstance(schema["properties"], dict)
            if "required" in schema:
                assert isinstance(schema["required"], list)
                assert all(isinstance(name, str) for name in schema["required"])


# ─────────────────────────────────────────────────────────────────────────────
# Round-trip stability
# ─────────────────────────────────────────────────────────────────────────────


class TestRoundTripStability:
    """custom → JSONSchema → custom yields equivalent semantics."""

    def test_required_typed_round_trip(self):
        original = {"path": str, "depth": int}
        roundtrip = _json_schema_to_custom(_custom_to_json_schema(original))
        assert roundtrip == original

    def test_optional_with_default_round_trip(self):
        original = {"path": str, "limit": (int, 10), "force": (bool, False)}
        roundtrip = _json_schema_to_custom(_custom_to_json_schema(original))
        assert roundtrip == original

    def test_optional_with_none_default_round_trip(self):
        # Custom: (int, None) → JSONSchema omits default → custom: (int, None)
        original = {"path": str, "limit": (int, None)}
        roundtrip = _json_schema_to_custom(_custom_to_json_schema(original))
        assert roundtrip == original

    def test_description_value_round_trip_to_typed_required(self):
        # Description-as-value loses the description string but preserves
        # required+typed dispatch semantics.
        original = {"context": "describe what you want"}
        roundtrip = _json_schema_to_custom(_custom_to_json_schema(original))
        # After round-trip, dispatch sees a required str (same dispatch
        # behavior as the original — both are required).
        assert roundtrip == {"context": str}

    def test_jsonschema_authored_round_trip_via_export(self):
        # When a tool authors JSONSchema, exporting and re-feeding it as
        # input_schema produces the same normalized custom format.
        t = _JsonSchemaTool()
        exported = t.to_json_schema()

        class _Reauthored(Tool):
            name = "reauthored"
            input_schema = exported

            def execute(self, **kwargs):
                return None

        t2 = _Reauthored()
        # Same dispatch semantics on both tools.
        assert t.input_schema == t2.input_schema

    def test_dispatch_equivalence_after_round_trip(self):
        # The strongest round-trip check: build a tool from each format
        # and verify dispatch behavior is identical for the same inputs.
        original_custom = {"a": str, "b": (int, 7)}

        class _CustomA(Tool):
            name = "a"
            input_schema = original_custom

            def execute(self, **kwargs):
                return kwargs

        class _JsonB(Tool):
            name = "b"
            input_schema = _custom_to_json_schema(original_custom)

            def execute(self, **kwargs):
                return kwargs

        # Both reject missing required.
        assert _CustomA().run().success is False
        assert _JsonB().run().success is False

        # Both accept the same inputs.
        r1 = _CustomA().run(a="hello")
        r2 = _JsonB().run(a="hello")
        assert r1.success is True and r2.success is True
        assert r1.output == r2.output


# ─────────────────────────────────────────────────────────────────────────────
# Negative + edge case checks
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_tool_without_name_raises(self):
        class _Anon(Tool):
            name = ""

            def execute(self, **kwargs):
                return None

        with pytest.raises(ValueError, match="non-empty name"):
            _Anon()

    def test_jsonschema_with_no_required_field(self):
        class _AllOptional(Tool):
            name = "all_opt"
            input_schema = {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "integer", "default": 0},
                },
            }

            def execute(self, **kwargs):
                return kwargs

        t = _AllOptional()
        # All params are optional — empty dict should validate.
        assert t.run().success is True

    def test_jsonschema_with_empty_properties(self):
        class _NoProps(Tool):
            name = "no_props"
            input_schema = {"type": "object", "properties": {}}

            def execute(self, **kwargs):
                return "ok"

        t = _NoProps()
        assert t.run().success is True
        assert t.to_json_schema() == {"type": "object", "properties": {}}

    def test_to_json_schema_with_non_dict_input_schema(self):
        # Defensive: if a tool subclass somehow has a non-dict
        # input_schema, to_json_schema should produce a safe empty schema
        # rather than crash.
        class _Weird(Tool):
            name = "weird"
            input_schema = "not a dict"  # type: ignore[assignment]

            def execute(self, **kwargs):
                return None

        t = _Weird()
        assert t.to_json_schema() == {"type": "object", "properties": {}}

    def test_run_returns_tooloutput_directly(self):
        # Confirms ToolOutput passthrough still works after CC9 changes.
        class _Direct(Tool):
            name = "direct"
            input_schema = {"x": str}

            def execute(self, **kwargs):
                return ToolOutput(success=True, output={"got": kwargs["x"]})

        result = _Direct().run(x="hi")
        assert isinstance(result, ToolOutput)
        assert result.success is True
        assert result.output == {"got": "hi"}


# ─────────────────────────────────────────────────────────────────────────────
# Regression: agent_loop dynamic-tool prompt builder works for BOTH formats
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentLoopParamRendering:
    """Pin that the dynamic-tool prompt builder in
    ``runtime/agent_loop.py`` (post-CC9, routes through ``to_json_schema()``)
    produces non-empty params for both authored formats.

    Pre-CC9 this iterated ``input_schema.items()`` directly. For
    custom-format tools that worked. For JSONSchema-authored tools
    (``@maxim.tool`` decorator), iterating ``.items()`` gave
    ``("type", "object")``, ``("properties", {...})``, etc., and the
    builder called ``.__name__`` on the value — the resulting
    ``AttributeError`` was swallowed by an outer
    ``except (KeyError, Exception): pass``, so JSONSchema-authored tools
    silently produced empty prompts. CC9 fixes this by routing through
    ``Tool.to_json_schema()``. This test pins both formats render correctly.
    """

    @staticmethod
    def _render_params_via_to_json_schema(tool: Tool) -> dict[str, str]:
        """Mirror the post-CC9 logic in agent_loop.py."""
        json_schema = tool.to_json_schema()
        properties = json_schema.get("properties", {}) or {}
        required = set(json_schema.get("required", []) or [])
        params: dict[str, str] = {}
        for name, prop in properties.items():
            prop_type = prop.get("type", "string") if isinstance(prop, dict) else "string"
            if name in required:
                params[name] = str(prop_type)
            else:
                default = prop.get("default") if isinstance(prop, dict) else None
                params[name] = f"({prop_type}, default={default!r})"
        return params

    def test_custom_format_tool_renders_non_empty(self):
        params = self._render_params_via_to_json_schema(_CustomFormatTool())
        assert "path" in params
        assert params["path"] == "string"
        # Optional with non-None default surfaces its default.
        assert "default=False" in params["overwrite"]

    def test_jsonschema_format_tool_renders_non_empty(self):
        # The pre-CC9 bug: this would silently be empty.
        params = self._render_params_via_to_json_schema(_JsonSchemaTool())
        assert params  # MUST be non-empty
        assert params["path"] == "string"
        assert "default=False" in params["overwrite"]

    def test_no_schema_tool_renders_empty(self):
        params = self._render_params_via_to_json_schema(_NoSchemaTool())
        assert params == {}

    def test_real_decorated_tool_renders_non_empty(self):
        """Integration: the @maxim.tool decorator authors JSONSchema. The
        dynamic-tool prompt builder must surface its params correctly.
        """
        from maxim.api import _pending_tools, tool

        @tool
        def my_thing(query: str, limit: int = 5) -> list:
            """test tool."""
            return []

        registered = _pending_tools[-1]
        params = self._render_params_via_to_json_schema(registered)
        assert "query" in params and "limit" in params
        assert params["query"] == "string"
        assert "default=5" in params["limit"]
