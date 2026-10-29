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
        from maxim.utils.internet_access import EffectiveInternetPolicy, InternetAccessPolicy

        policy = EffectiveInternetPolicy(policy=InternetAccessPolicy(), enabled=True)
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

    from maxim.utils.internet_access import EffectiveInternetPolicy, InternetAccessPolicy

    policy = EffectiveInternetPolicy(policy=InternetAccessPolicy(), enabled=True)
    tool = InternetSearchTool(get_internet_policy=lambda: policy)
    ret = ([], "boom") if failed else ([], None)
    with patch("maxim.tools.internet_search._search_duckduckgo", return_value=ret):
        out = tool.execute(query="q")
    assert bool(out.metadata.get("search_failed", False)) is failed


class TestHtmlScraperGoesThroughUtilsHttp:
    """The real ``_search_duckduckgo_lite``, not a patch of it: the urllib3 pool it used whenever urllib3
    was importable skipped utils/http (the #825 cap, the caller's timeout, the proxy settings), and its
    blanket except turned every failure into ``[]`` -- "nothing matched"."""

    _PAGE = (
        b'<a rel="nofollow" class="result__a" href="https://example.com/a">Example A</a>'
        b'<a class="result__snippet" href="https://example.com/a">An example snippet about A</a>'
    )

    def _search(self, scrape):
        from maxim.utils import http

        def fetch(url, **kwargs):
            if url.startswith("https://html.duckduckgo.com/"):
                return scrape(url, **kwargs)
            return http.Response(
                status=200, headers={}, content=b"{}", elapsed_ms=0.0, endpoint="_external", request_id="r"
            )

        with (
            patch("maxim.tools.internet_search._search_with_ddg_package", return_value=None),
            patch("maxim.utils.http.fetch_url", side_effect=fetch),
        ):
            return _search_duckduckgo("anything", timeout_s=4.0)

    def test_a_transport_failure_is_reported_not_swallowed(self):
        def boom(url, **kwargs):
            raise OSError("connection refused")

        results, failure = self._search(boom)
        assert results == []
        assert failure and "connection refused" in failure

    def test_a_redirect_wrapped_result_is_unwrapped_not_dropped(self):
        # DuckDuckGo's HTML page links each result through //duckduckgo.com/l/?uddg=<target>; the scraper filtered
        # duckduckgo.com links BEFORE unwrapping, so a normal results page read as "nothing matched".
        from maxim.utils import http

        page = (
            b'<a rel="nofollow" class="result__a" href="//duckduckgo.com/y.js?ad_domain=x">An ad</a>'
            b'<a rel="nofollow" class="result__a" '
            b'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fq%3Da%2526b&amp;rut=abc">Example A</a>'
            b'<a class="result__snippet" href="x">The snippet for example A</a>'
        )

        def wrapped(url, **kwargs):
            return http.Response(
                status=200, headers={}, content=page, elapsed_ms=0.0, endpoint="_external", request_id="r"
            )

        results, failure = self._search(wrapped)
        assert failure is None
        # Decoded once: the target's own %26 survives.
        assert [(r["url"], r["title"]) for r in results] == [("https://example.com/a?q=a%26b", "Example A")]

    def test_only_duckduckgos_own_host_is_filtered_never_a_lookalike(self):
        # The host check is exact (CodeQL py/incomplete-url-substring-sanitization on the first version, a substring
        # test): results hosted at a lookalike must be kept, DuckDuckGo's own links dropped.
        from maxim.tools.internet_search import _is_duckduckgo_host

        assert _is_duckduckgo_host("//duckduckgo.com/y.js?ad_domain=x")
        assert _is_duckduckgo_host("https://html.duckduckgo.com/settings")
        assert not _is_duckduckgo_host("https://notduckduckgo.com/a")
        assert not _is_duckduckgo_host("https://duckduckgo.com.example/a")
        assert not _is_duckduckgo_host("https://example.com/duckduckgo.com-review")

    def test_a_bot_challenge_202_is_a_failure_not_an_empty_result(self):
        # Captured live 2026-10-26: html.duckduckgo.com answered a scripted query with HTTP 202 and its
        # "anomaly" challenge page, which parses to zero links.
        from maxim.utils import http

        def challenge(url, **kwargs):
            return http.Response(
                status=202,
                headers={},
                content=b"<div class='anomaly-modal__title'>bots</div>",
                elapsed_ms=0.0,
                endpoint="_external",
                request_id="r",
            )

        results, failure = self._search(challenge)
        assert results == []
        assert failure and "HTTP 202" in failure

    def test_a_page_is_fetched_bounded_with_the_callers_timeout(self):
        from maxim.utils import http

        seen = {}

        def page(url, **kwargs):
            seen.update(kwargs)
            return http.Response(
                status=200, headers={}, content=self._PAGE, elapsed_ms=0.0, endpoint="_external", request_id="r"
            )

        results, failure = self._search(page)
        assert failure is None
        assert [r["url"] for r in results] == ["https://example.com/a"]
        assert seen["max_bytes"] == 2_000_000
        assert seen["timeout"].read_s == 4.0
