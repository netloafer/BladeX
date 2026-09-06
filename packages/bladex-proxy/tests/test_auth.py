"""客户端 key 验证单元测试。"""

from bladex_proxy.auth import KeyStore
from bladex_proxy.config import ProxyConfig


class TestKeyStore:
    def test_parse_empty(self):
        ks = KeyStore.parse("")
        assert len(ks.keys) == 0

    def test_parse_single_key_no_label(self):
        ks = KeyStore.parse("bladex-abc123")
        assert len(ks.keys) == 1
        assert ks.keys[0].key == "bladex-abc123"
        assert ks.keys[0].label == ""

    def test_parse_multiple_keys_with_labels(self):
        ks = KeyStore.parse("bladex-abc123||jason-macbook||bladex-def456||jason-codex")
        assert len(ks.keys) == 2
        assert ks.keys[0].key == "bladex-abc123"
        assert ks.keys[0].label == "jason-macbook"
        assert ks.keys[1].key == "bladex-def456"
        assert ks.keys[1].label == "jason-codex"

    def test_parse_key_without_label_in_middle(self):
        """key 后面没有 label（下一条是另一个 key）。"""
        ks = KeyStore.parse("bladex-aaa||label-a||bladex-bbb")
        assert len(ks.keys) == 2
        assert ks.keys[0].key == "bladex-aaa"
        assert ks.keys[0].label == "label-a"
        assert ks.keys[1].key == "bladex-bbb"
        assert ks.keys[1].label == ""

    def test_verify_valid_key(self):
        ks = KeyStore.parse("bladex-abc123||jason-macbook")
        ok, reason = ks.verify("bladex-abc123")
        assert ok is True
        assert reason == "jason-macbook"

    def test_verify_invalid_key(self):
        ks = KeyStore.parse("bladex-abc123||jason-macbook")
        ok, reason = ks.verify("wrong-key")
        assert ok is False
        assert reason == "invalid_key"

    def test_verify_none_key(self):
        ks = KeyStore.parse("bladex-abc123")
        ok, reason = ks.verify(None)
        assert ok is False
        assert reason == "missing_key"

    def test_verify_empty_keystore(self):
        ks = KeyStore.parse("")
        ok, reason = ks.verify("any-key")
        assert ok is False
        assert reason == "invalid_key"

    def test_labels_property(self):
        ks = KeyStore.parse("bladex-abc||jason-macbook||bladex-def")
        labels = ks.labels
        assert "jason-macbook" in labels
        assert len(labels) == 2


class TestProxyConfigAuth:
    def test_auth_disabled_by_default(self):
        """默认不校验。"""
        config = ProxyConfig()
        assert config.auth_enabled is False

    def test_auth_enabled(self):
        config = ProxyConfig(auth_enabled=True, client_keys_raw="bladex-abc||jason")
        assert config.auth_enabled is True
        assert len(config.key_store.keys) == 1

    def test_auth_check_disabled_passes_anything(self):
        """关闭校验时，任意 key（含 None）都放行。"""
        config = ProxyConfig(auth_enabled=False)
        ok, reason = config.auth_check(None)
        assert ok is True
        assert reason == "auth_disabled"

        ok, reason = config.auth_check("anything")
        assert ok is True

    def test_auth_check_enabled_valid_key(self):
        config = ProxyConfig(
            auth_enabled=True,
            client_keys_raw="bladex-abc123||jason-macbook",
        )
        ok, reason = config.auth_check("bladex-abc123")
        assert ok is True
        assert reason == "jason-macbook"

    def test_auth_check_enabled_invalid_key(self):
        config = ProxyConfig(
            auth_enabled=True,
            client_keys_raw="bladex-abc123",
        )
        ok, reason = config.auth_check("wrong-key")
        assert ok is False
        assert reason == "invalid_key"

    def test_auth_check_enabled_missing_key(self):
        config = ProxyConfig(
            auth_enabled=True,
            client_keys_raw="bladex-abc123",
        )
        ok, reason = config.auth_check(None)
        assert ok is False
        assert reason == "missing_key"
