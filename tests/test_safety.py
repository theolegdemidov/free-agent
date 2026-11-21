import importlib
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def load_agent(workspace, **env):
    with mock.patch.dict(os.environ, {"AGENT_WORKSPACE": str(workspace), **env}):
        import agent
        return importlib.reload(agent)


class TestFileSafety(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = pathlib.Path(self.tmp.name)
        self.agent = load_agent(self.ws)

    def tearDown(self):
        self.tmp.cleanup()

    def test_history_file_is_protected(self):
        for name in [".history.json", ".history.json.tmp", "sub/.history.json"]:
            with self.assertRaises(ValueError):
                self.agent.write_file(name, "{}")

    def test_custom_history_path_is_protected(self):
        agent = load_agent(self.ws, AGENT_HISTORY=str(self.ws / "chat-log.json"))
        with self.assertRaises(ValueError):
            agent.write_file("chat-log.json", "[]")

    def test_size_limit(self):
        agent = load_agent(self.ws, AGENT_MAX_FILE_BYTES="10")
        agent.write_file("ok.txt", "x" * 10)
        with self.assertRaises(ValueError):
            agent.write_file("big.txt", "x" * 11)
        self.assertFalse((self.ws / "big.txt").exists())

    def reply(self, name="a.txt", body="новое"):
        return f"<<<FILE {name}>>>\n{body}\n<<<END>>>"

    def test_auto_yes_does_not_overwrite_existing(self):
        (self.ws / "a.txt").write_text("старое", encoding="utf-8")
        self.agent.handle_reply(self.reply(), auto_yes=True)
        self.assertEqual((self.ws / "a.txt").read_text(encoding="utf-8"), "старое")

    def test_overwrite_flag_allows_replacement(self):
        (self.ws / "a.txt").write_text("старое", encoding="utf-8")
        self.agent.handle_reply(self.reply(), auto_yes=True, overwrite=True)
        self.assertEqual((self.ws / "a.txt").read_text(encoding="utf-8"), "новое")

    def test_new_file_written_with_auto_yes(self):
        self.agent.handle_reply(self.reply("n/new.txt"), auto_yes=True)
        self.assertEqual((self.ws / "n" / "new.txt").read_text(encoding="utf-8"), "новое")

    def test_interactive_asks_separately_for_existing(self):
        (self.ws / "a.txt").write_text("старое", encoding="utf-8")
        with mock.patch("builtins.input", return_value="n") as ask:
            self.agent.handle_reply(self.reply())
        self.assertIn("уже существует", ask.call_args.args[0])
        self.assertEqual((self.ws / "a.txt").read_text(encoding="utf-8"), "старое")
        with mock.patch("builtins.input", return_value="y"):
            self.agent.handle_reply(self.reply())
        self.assertEqual((self.ws / "a.txt").read_text(encoding="utf-8"), "новое")

    def test_blocked_path_in_reply_is_skipped_not_crashing(self):
        self.agent.handle_reply(self.reply("../evil.txt"), auto_yes=True)
        self.assertFalse((self.ws.parent / "evil.txt").exists())


class TestChat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.agent = load_agent(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def resp(self, code, body=None, headers=None):
        r = mock.Mock(status_code=code, ok=code < 400, text=str(body), headers=headers or {})
        r.json.return_value = body
        return r

    def test_retry_then_success(self):
        ok = self.resp(200, {"choices": [{"message": {"content": "привет"}}]})
        with mock.patch.object(self.agent.requests, "post", side_effect=[self.resp(429, {}, {"Retry-After": "0"}), ok]), \
                mock.patch.object(self.agent.time, "sleep"):
            self.assertEqual(self.agent.chat([]), "привет")

    def test_auth_error_is_friendly(self):
        with mock.patch.object(self.agent.requests, "post", return_value=self.resp(401, {})):
            with self.assertRaises(self.agent.AgentError):
                self.agent.chat([])

    def test_gives_up_after_retries(self):
        with mock.patch.object(self.agent.requests, "post", return_value=self.resp(503, {})), \
                mock.patch.object(self.agent.time, "sleep"):
            with self.assertRaises(self.agent.AgentError):
                self.agent.chat([])

    def test_bad_response_shape(self):
        with mock.patch.object(self.agent.requests, "post", return_value=self.resp(200, {"oops": 1})):
            with self.assertRaises(self.agent.AgentError):
                self.agent.chat([])


if __name__ == "__main__":
    unittest.main()
