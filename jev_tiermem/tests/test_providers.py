"""Exercise real SDK serialization over mocked HTTP, never over the network."""

import importlib.util
import json
import unittest
from unittest.mock import patch

from jev_tiermem import Config
from jev_tiermem.providers import JevHTTPClient, JevRouter, OpenAIModel


@unittest.skipUnless(importlib.util.find_spec("typesafe_sdk"), "optional live extra is not installed")
class JevSDKTests(unittest.TestCase):
    def test_configured_decide_endpoint_auth_model_and_usage(self):
        import httpx2

        def handle(request):
            self.assertEqual(str(request.url), "https://example.invalid/api/v1/decide")
            self.assertEqual(request.headers["Authorization"], "Bearer test-only")
            self.assertEqual(json.loads(request.content)["model"], "jev-1.13.0")
            return httpx2.Response(200, json={
                "model": "jev-1.13.0", "answers": {"sufficient": {"type": "noul", "noul": 0.96}},
                "usage": {"input_tokens": 288, "output_tokens": 23},
            })

        config = Config(jev_model="jev-1.13.0", jev_api_url="https://example.invalid/api/v1/decide")
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-only"}), \
                httpx2.Client(transport=httpx2.MockTransport(handle)) as http:
            router = JevRouter(config, client=JevHTTPClient(config, client=http))
            self.assertTrue(router.sufficient("timeout?", [{"text": "83 seconds"}]).accepted)
        self.assertEqual(router.models_seen, {"jev-1.13.0"})
        self.assertEqual((router.input_tokens, router.output_tokens), (288, 23))

    def test_configured_endpoint_auth_error_fails_closed_without_body(self):
        import httpx2

        config = Config(jev_api_url="https://example.invalid/api/v1/decide")
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-only"}), \
                httpx2.Client(transport=httpx2.MockTransport(
                    lambda request: httpx2.Response(401, json={"detail": "sensitive server message"})
                )) as http:
            router = JevRouter(config, client=JevHTTPClient(config, client=http))
            result = router.sufficient("timeout?", [{"text": "83 seconds"}])
        self.assertFalse(result.accepted)
        self.assertEqual(result.error, "HTTPStatusError")
        self.assertNotIn("sensitive", str(result))

    def test_official_sdk_request_response_and_usage(self):
        import httpx2
        from typesafe_sdk import RetryPolicy, TypeSafeClient

        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            self.assertEqual(request.url.path, "/v1/systemone")
            return httpx2.Response(200, json={
                "model": "jev-latest",
                "answers": {"sufficient": {"type": "noul", "noul": 0.96}},
                "usage": {"input_tokens": 37, "output_tokens": 1},
            })

        with TypeSafeClient(api_key="test-only", model="jev-latest", retry=RetryPolicy(max_retries=0),
                            transport=httpx2.MockTransport(handle)) as client:
            router = JevRouter(Config(), client=client)
            decision = router.sufficient("timeout?", [{"id": "s-one", "text": "60 seconds"}])
        self.assertTrue(decision.accepted, decision.error)
        self.assertEqual(requests[0]["model"], "jev-latest")
        self.assertEqual(requests[0]["questions"]["sufficient"]["type"], "noul")
        self.assertEqual((router.calls, router.input_tokens, router.output_tokens), (1, 37, 1))


@unittest.skipUnless(importlib.util.find_spec("openai"), "optional live extra is not installed")
class TextSDKTests(unittest.TestCase):
    def test_text_model_json_and_usage(self):
        import openai
        # OpenAI 3 uses httpx2; OpenAI 1/2 use httpx. Match the installed SDK's client.
        import importlib
        http = importlib.import_module("httpx2" if int(openai.__version__.split('.')[0]) >= 3 else "httpx")
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            self.assertEqual(request.url.path, "/v1/chat/completions")
            return http.Response(200, json={
                "id": "test", "object": "chat.completion", "created": 0, "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": '{"summary":"60 seconds"}'}}],
                "usage": {"prompt_tokens": 41, "completion_tokens": 7, "total_tokens": 48},
            })

        with openai.OpenAI(api_key="test-only", base_url="https://example.invalid/v1", max_retries=0,
                           http_client=openai.DefaultHttpxClient(transport=http.MockTransport(handle))) as client:
            model = OpenAIModel(Config(model="test-model"), client=client)
            result = model.complete("summarize", "Return a summary", {"raw": ["60 seconds"]})
        self.assertEqual(result, {"summary": "60 seconds"})
        self.assertEqual(requests[0]["response_format"], {"type": "json_object"})
        self.assertEqual((model.calls, model.input_tokens, model.output_tokens), (1, 41, 7))


if __name__ == "__main__":
    unittest.main()
