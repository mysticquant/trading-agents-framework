from pydantic import BaseModel

from trading_agents.llm.structured_output import model_to_tool_schema, parse_tool_input


class _Nested(BaseModel):
    label: str
    weight: float


class _Outer(BaseModel):
    name: str
    items: list[_Nested]


def test_model_to_tool_schema_inlines_nested_defs():
    tool = model_to_tool_schema(_Outer)
    assert tool["name"] == "emit_result"
    assert "$defs" not in tool["input_schema"]
    items_schema = tool["input_schema"]["properties"]["items"]
    assert "$ref" not in str(items_schema)
    assert items_schema["items"]["properties"]["label"]["type"] == "string"


def test_parse_tool_input_round_trips():
    payload = {"name": "cycle-1", "items": [{"label": "a", "weight": 1.5}]}
    parsed = parse_tool_input(payload, _Outer)
    assert parsed.name == "cycle-1"
    assert parsed.items[0].label == "a"
    assert parsed.items[0].weight == 1.5
