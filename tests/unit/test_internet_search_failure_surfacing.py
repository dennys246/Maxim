"""A broken web search must reach the agent, not vanish into a log line.

Reported from live console use: a question triggered internet_search, the
HTTP layer logged `http_request_failed`, and `execute` returned
`success=True, output=[], "No results found"` — indistinguishable from a
search that genuinely matched nothing. The agent had no signal it had
failed, so it said nothing, and the user got 163 seconds of silence.

This is the "graceful degraded states, SHOWN — never a silent wedge" rule.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from maxim.tools.internet_search import _search_duckduckgo


class TestHelperReportsWhy:
    def test_transport_failure_returns_a_reason(self):
        # Every method fails → the helper must say WHY, not just return [].
        with (
            patch("maxim.tools.internet_search._search_with_ddg_package", return_value=None),
            patch("maxim.tools.internet_search._search_duckduckgo_lite", side_effect=OSError("connection refused")),
            patch("maxim.utils.http.fetch_url", side_effect=OSError("connection refused")),
        ):
            results, failure = _search_duckduckgo("anything")
        assert results == []
        # Don't over-pin the wording (the last handler to fire owns it) — pin
        # that a reason EXISTS and names the underlying cause, which is what
        # the agent needs in order to tell the user something true.
        assert failure
        assert "connection refused" in failure

    def test_genuine_empty_is_not_a_failure(self):
        # The distinction that did not exist before: nothing matched, but the
        # search itself worked.
        with (
            patch("maxim.tools.internet_search._search_with_ddg_package", return_value=[]),
        ):
            results, failure = _search_duckduckgo("anything")
        assert results == []
        assert failure is None

    def test_success_returns_no_failure(self):
        hit = [{"title": "t", "url": "https://example.com", "snippet": "s"}]
        with patch("maxim.tools.internet_search._search_with_ddg_package", return_value=hit):
            results, failure = _search_duckduckgo("anything")
        assert results == hit and failure is None


class TestExecuteSurfacesTheFailure:
    def _tool(self):
        from maxim.tools.internet_search import InternetSearchTool

        # execute() blocks by default without a policy; give it a permissive one.
        policy = type("P", (), {"enabled": True, "request_timeout_s": 8.0})()
        return InternetSearchTool(get_internet_policy=lambda: policy)

    def test_broken_search_is_reported_as_a_failure(self):
        tool = self._tool()
        with patch("maxim.tools.internet_search._search_duckduckgo", return_value=([], "the provider timed out")):
            out = tool.execute(query="weather")
        assert out.success is False, "a broken search must not report success"
        assert "failed" in (out.error or "").lower()
        assert "timed out" in (out.error or "")
        # The agent needs an instruction it can act on, not just a code.
        assert "tell the user" in (out.error or "").lower()

    def test_empty_but_working_search_still_succeeds(self):
        # Must NOT regress into treating "no matches" as an error.
        tool = self._tool()
        with patch("maxim.tools.internet_search._search_duckduckgo", return_value=([], None)):
            out = tool.execute(query="asdkjhasd")
        assert out.success is True
        assert out.metadata.get("result_count") == 0
        assert "search itself worked" in str(out.metadata.get("message", ""))

    def test_results_path_unchanged(self):
        tool = self._tool()
        hit = [{"title": "t", "url": "https://example.com", "snippet": "s"}]
        with patch("maxim.tools.internet_search._search_duckduckgo", return_value=(hit, None)):
            out = tool.execute(query="weather")
        assert out.success is True
        assert out.metadata["result_count"] == 1


@pytest.mark.parametrize("failed", [True, False])
def test_failure_flag_is_in_metadata(failed):
    # The console/bio panel reads metadata; the flag makes the failure
    # machine-checkable rather than string-matched.
    from maxim.tools.internet_search import InternetSearchTool

    policy = type("P", (), {"enabled": True, "request_timeout_s": 8.0})()
    tool = InternetSearchTool(get_internet_policy=lambda: policy)
    ret = ([], "boom") if failed else ([], None)
    with patch("maxim.tools.internet_search._search_duckduckgo", return_value=ret):
        out = tool.execute(query="q")
    assert bool(out.metadata.get("search_failed", False)) is failed
