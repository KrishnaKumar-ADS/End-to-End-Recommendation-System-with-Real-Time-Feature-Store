"""Unit tests for webhook signature verification helper."""

from __future__ import annotations

import hashlib
import hmac
from unittest.mock import patch

from api.routes.webhook import _verify_github_signature


class TestVerifyGithubSignature:
    def test_no_secret_always_accepts(self):
        with patch("api.routes.webhook.get_settings") as mock_s:
            mock_s.return_value.webhook_secret = ""
            assert _verify_github_signature(b"body", "sha256=anything") is True

    def test_no_secret_accepts_even_without_header(self):
        with patch("api.routes.webhook.get_settings") as mock_s:
            mock_s.return_value.webhook_secret = ""
            assert _verify_github_signature(b"body", None) is True

    def test_valid_signature_accepted(self):
        secret = "test_secret_key"
        body = b'{"action": "push"}'
        sig = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()

        with patch("api.routes.webhook.get_settings") as mock_s:
            mock_s.return_value.webhook_secret = secret
            assert _verify_github_signature(body, sig) is True

    def test_invalid_signature_rejected(self):
        secret = "test_secret_key"
        body = b'{"action": "push"}'

        with patch("api.routes.webhook.get_settings") as mock_s:
            mock_s.return_value.webhook_secret = secret
            assert _verify_github_signature(body, "sha256=wrongsignature") is False

    def test_missing_header_rejected_when_secret_set(self):
        with patch("api.routes.webhook.get_settings") as mock_s:
            mock_s.return_value.webhook_secret = "secret123"
            assert _verify_github_signature(b"body", None) is False

    def test_header_without_sha256_prefix_rejected(self):
        with patch("api.routes.webhook.get_settings") as mock_s:
            mock_s.return_value.webhook_secret = "secret123"
            assert _verify_github_signature(b"body", "md5=abc123") is False
