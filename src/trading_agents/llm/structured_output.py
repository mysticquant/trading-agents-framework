"""Turns a pydantic model into an Anthropic tool schema, and parses a tool_use
block's input back into that model.

Pydantic emits nested models as `$defs` + `$ref` in its JSON schema; the
Anthropic tool `input_schema` must be a single self-contained object schema,
so nested `$defs` are inlined here rather than left as refs.
"""

from typing import TypeVar, cast

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

TOOL_NAME = "emit_result"


def model_to_tool_schema(output_model: type[BaseModel]) -> dict:
    """Build an Anthropic `tool` definition whose `input_schema` matches
    `output_model`, with any `$defs` inlined."""
    schema = output_model.model_json_schema()
    defs = schema.pop("$defs", {})
    if defs:
        schema = cast(dict, _inline_refs(schema, defs))
    return {
        "name": TOOL_NAME,
        "description": f"Emit a {output_model.__name__} result.",
        "input_schema": schema,
    }


def _inline_refs(node: object, defs: dict) -> object:
    if isinstance(node, dict):
        if "$ref" in node:
            ref_name = node["$ref"].rsplit("/", 1)[-1]
            return _inline_refs(defs[ref_name], defs)
        return {key: _inline_refs(value, defs) for key, value in node.items()}
    if isinstance(node, list):
        return [_inline_refs(item, defs) for item in node]
    return node


def parse_tool_input(tool_input: dict, output_model: type[T]) -> T:
    """Validate a tool_use block's `input` dict into `output_model`."""
    return output_model.model_validate(tool_input)
