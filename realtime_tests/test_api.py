import unittest
from fastapi.testclient import TestClient
from app.core.config import Settings
from app.main import create_app
from app.translation.backend import TranslationResult


class FakeBackend:
    async def translate(self, source, *args, **kwargs):
        return TranslationResult("번역:" + source, 1)

    async def close(self):
        pass


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(_env_file=None, streaming_api_key="test-admin-" + "x" * 32,
                                 translation_commit_delay_ms=0, translation_debounce_ms=0)
        self.client = TestClient(create_app(self.settings, backend=FakeBackend()))
        self.client.__enter__()
        self.admin = {"Authorization": "Bearer " + self.settings.streaming_api_key.get_secret_value()}

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def create(self):
        result = self.client.post("/sessions", headers=self.admin, json={})
        self.assertEqual(result.status_code, 201, result.text)
        data = result.json()
        return data, {"Authorization": "Bearer " + data["session_token"]}

    def test_health_admin_and_session_capability_auth(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.client.post("/sessions", json={}).status_code, 401)
        data, headers = self.create()
        path = "/sessions/" + data["session_id"]
        self.assertEqual(self.client.get(path, headers=self.admin).status_code, 404)
        self.assertEqual(self.client.get(path, headers=headers).status_code, 200)
        self.assertEqual(self.client.get("/metrics").status_code, 401)
        self.assertEqual(self.client.get("/metrics", headers=self.admin).status_code, 200)
        self.assertEqual(self.client.delete(path, headers=headers).status_code, 204)
        self.assertEqual(self.client.get(path, headers=headers).status_code, 404)

    def test_websocket_final_and_recoverable_invalid_json(self):
        data, headers = self.create()
        with self.client.websocket_connect(data["websocket_path"], headers=headers) as websocket:
            self.assertEqual(websocket.receive_json()["type"], "session_state")
            websocket.send_text("{")
            self.assertEqual(websocket.receive_json()["code"], "INVALID_MESSAGE")
            websocket.send_json({"type": "asr_final", "sequence": 0, "text": "Please stop."})
            events = []
            for _ in range(10):
                item = websocket.receive_json()
                events.append(item)
                if item.get("final"):
                    break
                self.assertNotEqual(item["type"], "error", item)
            self.assertTrue(events[-1]["final"])
            self.assertEqual(events[-1]["draft"], "")
            websocket.send_json({"type": "ping"})
            self.assertEqual(websocket.receive_json()["type"], "pong")

    def test_browser_first_frame_auth_and_session_isolation(self):
        data, _ = self.create()
        with self.client.websocket_connect(data["websocket_path"]) as websocket:
            websocket.send_json({"type": "authenticate", "token": data["session_token"]})
            self.assertEqual(websocket.receive_json()["session_id"], data["session_id"])
        other, other_headers = self.create()
        self.assertEqual(self.client.get("/sessions/" + data["session_id"], headers=other_headers).status_code, 404)

    def test_body_limit_model_allowlist_glossary_and_configuration(self):
        self.assertEqual(self.client.post("/sessions", content=b"x" * 65537, headers=self.admin).status_code, 413)
        self.assertEqual(self.client.post("/sessions", json={"translation": {"model": "arbitrary-model"}}, headers=self.admin).status_code, 422)
        data, headers = self.create()
        path = "/sessions/" + data["session_id"] + "/glossary"
        self.assertEqual(self.client.post(path, json={"entries": {"CUDA": "CUDA"}}, headers=headers).status_code, 200)
        self.assertEqual(self.client.post(path, json={"entries": {"": "x"}}, headers=headers).status_code, 422)
        self.assertEqual(self.client.post(path, json={"entries": {"CUDA": "CUDA"}}).status_code, 404)
