"""Regression tests for openai_backend._sanitize_error_message.

Covers the Cloudflare-502-HTML-flood bug: the backend used to log
`str(last_err)` verbatim, which on a tunneled 502 meant ~4KB of HTML in
a single warning line. The sanitizer now collapses HTML bodies to a
one-line summary and truncates any plain-text message over max_len.
"""

from __future__ import annotations

from maxim.models.language.openai_backend import _sanitize_error_message


class TestSanitizeErrorMessage:
    def test_none_returns_empty(self):
        assert _sanitize_error_message(None) == ""

    def test_short_plain_message_unchanged(self):
        err = RuntimeError("Connection refused")
        assert _sanitize_error_message(err) == "Connection refused"

    def test_long_plain_message_truncated(self):
        err = RuntimeError("x" * 2000)
        result = _sanitize_error_message(err, max_len=500)
        assert result.endswith("… (truncated)")
        assert len(result) <= 500 + len("… (truncated)")

    def test_html_doctype_collapsed_with_title(self):
        html_body = (
            "<!DOCTYPE html>\n"
            "<html><head><title>502 Bad Gateway</title></head>"
            "<body><h1>502 Bad Gateway</h1>"
            "<p>The server encountered a gateway error...</p>"
            + ("x" * 3000)  # simulate real bulk
            + "</body></html>"
        )
        err = RuntimeError(html_body)
        result = _sanitize_error_message(err)
        assert "<502 Bad Gateway," in result
        assert "bytes>" in result
        assert "<!DOCTYPE" not in result
        assert "<html" not in result.lower()

    def test_html_without_title_uses_fallback_label(self):
        html_body = "<!DOCTYPE html>\n<html><body>no title here</body></html>"
        err = RuntimeError(html_body)
        result = _sanitize_error_message(err)
        assert "<HTML error body," in result
        assert "bytes>" in result

    def test_html_after_exception_chain_prefix(self):
        """Python 3.11+ can wrap exceptions so str(err) starts with the
        outer exception's label before the inner message. The sanitizer
        must detect HTML anywhere in the first 200 chars, not just at
        position 0 after lstrip."""
        chained = (
            "RuntimeError caught during LLM call: "
            "<!DOCTYPE html>\n<html><head><title>502 Bad Gateway</title>"
            "</head><body>" + ("x" * 2000) + "</body></html>"
        )
        err = RuntimeError(chained)
        result = _sanitize_error_message(err)
        assert "<502 Bad Gateway," in result
        assert len(result) < 100  # much shorter than the 2KB original

    def test_html_upper_case_tag(self):
        err = RuntimeError("<HTML><BODY>error</BODY></HTML>" + "x" * 1000)
        result = _sanitize_error_message(err)
        assert "bytes>" in result
        assert len(result) < 100

    def test_baseexception_keyboardinterrupt_handled(self):
        """_sanitize_error_message accepts BaseException so it doesn't
        crash if called with a KeyboardInterrupt (which inherits from
        BaseException, not Exception)."""
        err = KeyboardInterrupt()
        # Should return whatever str(KeyboardInterrupt()) gives (empty
        # by default) without raising.
        result = _sanitize_error_message(err)
        assert isinstance(result, str)
