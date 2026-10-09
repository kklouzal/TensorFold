"""Malformed Responses objects refuse before chat; valid translations preserve content."""
import json
import unittest

from tests.http_fakes import post


class App:
    served_name = "shape-fixture"
    model_ids = [served_name]
    max_batch_size = 1
    exact_mode = {"mode": "exact"}

    def __init__(self):
        self.calls = []

    def chat(self, messages, *, max_tokens=None, temperature=0, on_delta=None, tools=None, **kwargs):
        self.calls.append((messages, tools))
        return {"content": "preserved", "finish_reason": "stop", "prompt_tokens": 3,
                "completion_tokens": 1, "cached_tokens": 0}


class ResponseShapeTests(unittest.TestCase):
    def refused(self, extra):
        app = App()
        status, raw = post(app, {"input": "Keep my input.", **extra}, path="/v1/responses")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(raw))
        self.assertEqual(app.calls, [])

    def test_nonobject_format_is_a_400_without_model_work(self):
        for value in ("text", [], ["text"], 1, False):
            with self.subTest(value=value):
                self.refused({"text": {"format": value}})

    def test_invalid_allowed_tool_shapes_are_400_without_model_work(self):
        for tools in (False, 1, "function", {}, [{"type": "function", "name": []}],
                      [{"type": "function", "name": {}}], [{"type": "function"}],
                      [{"type": "function", "name": ""}]):
            with self.subTest(tools=tools):
                self.refused({"tool_choice": {"type": "allowed_tools", "tools": tools}})

    def test_text_format_and_named_allowed_tools_preserve_translated_content(self):
        app = App()
        body = {"input": "Keep my input.", "text": {"format": {"type": "text"}},
                "tools": [{"type": "function", "name": "chosen"}, {"type": "function", "name": "other"}],
                "tool_choice": {"type": "allowed_tools", "tools": [{"type": "function", "name": "chosen"}]}}
        status, raw = post(app, body, path="/v1/responses")
        self.assertEqual(status, 200)
        reply = json.loads(raw)
        self.assertEqual(reply["output"][0]["content"][0]["text"], "preserved")
        self.assertEqual(app.calls[0][0], [{"role": "user", "content": "Keep my input."}])
        self.assertEqual([tool["function"]["name"] for tool in app.calls[0][1]], ["chosen"])


if __name__ == "__main__":
    unittest.main()
