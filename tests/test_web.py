"""sah.tools.web — the SSRF guard (the reason this module is careful), the declared-loss reporting, and
the parsers. No live network: the address checks run on IP literals (which resolve instantly, no DNS),
and every path that would open a socket patches the single `_get`/`_OPENER` seam."""
import gzip
import zlib
import urllib.error

import pytest

from sah import ingest
from sah.tools import web


# --- the address guard: fetch reaches the public internet, nothing else -----------------------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1:9090/admin",         # the model server itself
    "http://localhost/",                   # resolved to loopback below
    "http://10.1.2.3/",                    # private (RFC1918)
    "http://192.168.0.10:9090/",           # the LAN (RFC1918)
    "http://172.16.0.1/",                  # private
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata (link-local)
    "http://[::1]/",                       # IPv6 loopback
])
def test_fetch_guard_refuses_non_public_addresses(url):
    """These are IP literals: `getaddrinfo` answers them instantly with no network, so the guard is
    exercised for real, not stubbed. Each must be refused, and refused *as* a non-public address."""
    with pytest.raises(web._Blocked) as exc:
        web._guard(url, allow_private=False)
    if "169.254" in url or "127.0.0.1" in url or "10.1.2.3" in url or "192.168" in url \
            or "172.16" in url or "::1" in url:
        assert "not a public address" in str(exc.value)


@pytest.mark.parametrize("url", ["http://8.8.8.8/", "https://1.1.1.1/x", "https://93.184.216.34/"])
def test_fetch_guard_allows_public_addresses(url):
    web._guard(url, allow_private=False)      # public IP literals -> no raise, no network


def test_localhost_name_is_resolved_and_refused(monkeypatch):
    """A hostname, not a literal: the guard checks the resolved address, so `localhost` is caught by
    what it resolves to, not by a name blocklist that a `127.0.0.1.nip.io` would slip past."""
    monkeypatch.setattr(web.socket, "getaddrinfo",
                        lambda *a, **k: [(2, 1, 6, "", ("127.0.0.1", 80))])
    with pytest.raises(web._Blocked):
        web._guard("http://evil.example/", allow_private=False)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "data:text/plain,hi", "ftp://x/", "gopher://x/"])
def test_only_http_and_https(url):
    with pytest.raises(web._Blocked):
        web._guard(url, allow_private=False)   # scheme allowlist: no file://, no data://


def test_widening_opens_the_guard_but_not_the_scheme():
    web._guard("http://127.0.0.1:9090", allow_private=True)          # LAN widening -> allowed
    # Widening says an address is allowed, not that the socket may resolve whatever it likes: the
    # widened path is pinned to the address that was resolved, so the hole does not reopen behind the
    # operator who said yes.
    assert web._guard("http://127.0.0.1:9090", allow_private=True) == "127.0.0.1"
    with pytest.raises(web._Blocked):
        web._guard("file:///etc/passwd", allow_private=True)         # ...but never a scheme bypass


# --- fetch: refusal before a socket; declared loss on success ---------------------------------

def test_fetch_refuses_a_private_target_before_opening_anything():
    # 127.0.0.1 is an IP literal: the guard fires and returns a readable refusal; no socket, no DNS.
    report = web.Net(allow_private=False).fetch("http://127.0.0.1:9090/admin")
    assert report.startswith("web_fetch refused") and "not a public address" in report


class _Resp:
    def __init__(self, body, ctype):
        self._b, self.headers = body, {"Content-Type": ctype}

    def read(self, n=-1):
        return self._b if n is None or n < 0 else self._b[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FixedOpener:
    """Stand in for the socket: always serve one canned response. Lets a test drive the *real* `_get`
    (its guard, redirect loop, decompress) without a network -- stubbing `_get` itself would hide
    exactly the guard decision these tests exist to check."""

    def __init__(self, body, ctype):
        self._resp_args = (body, ctype)

    def open(self, req, timeout=None):
        return _Resp(*self._resp_args)


# --- the address that was checked is the address that is dialled ---------------------------------

def test_the_guard_hands_back_the_address_it_approved():
    """The check used to end with nothing, which left the connection free to resolve the name again and
    believe a different answer. The approved address is now the guard's output, because it is the only
    thing the socket may legitimately go to."""
    assert web._guard("http://8.8.8.8/x", allow_private=False) == "8.8.8.8"
    assert web._guard("https://93.184.216.34/", allow_private=False) == "93.184.216.34"


def test_a_pinned_connection_dials_the_approved_address_not_the_name(monkeypatch):
    """The whole of the fix at the one place a name could turn into a different machine: `create_connection`
    is handed an IP literal, so nothing resolves on the way to the socket."""
    dialed = []

    class Sock:
        def setsockopt(self, *a):        # connect() sets TCP_NODELAY and tolerates only OSError
            pass

    monkeypatch.setattr(web.socket, "create_connection",
                        lambda addr, *a, **k: dialed.append(addr) or Sock())
    web._PIN.ip = "93.184.216.34"
    try:
        conn = web._PinnedHTTPConnection("rebind.attacker.tld", port=80)
        conn.connect()
    finally:
        web._PIN.ip = ""
    assert dialed == [("93.184.216.34", 80)], "the socket goes where the guard looked"
    assert conn.host == "rebind.attacker.tld", "...and still calls itself the name it was given"


def test_a_pinned_tls_connection_still_names_the_host_it_is_verifying(monkeypatch):
    """Same pin for https, and the name survives: the stdlib's own `connect()` wraps the socket it was
    given and verifies against `self.host`, so pinning an address cannot be used to accept a certificate
    for something else. The pin is installed as an instance attribute because that is the attribute
    `http.client` calls -- a class-level override of the same name is shadowed, which is not obvious and
    cost one version of this fix its only real effect."""
    dialed = []
    monkeypatch.setattr(web.socket, "create_connection",
                        lambda addr, *a, **k: dialed.append(addr) or object())
    conn = web._PinnedHTTPSConnection("example.org")
    web._PIN.ip = "93.184.216.34"
    try:
        conn._create_connection(("example.org", 443), 5.0)      # what connect() calls
    finally:
        web._PIN.ip = ""
    assert dialed == [("93.184.216.34", 443)]
    assert conn.host == "example.org"


def test_a_name_that_moves_underneath_the_check_still_dials_the_checked_address(monkeypatch):
    """DNS rebinding, the shape the review named: answer a public address to the check and `127.0.0.1` to
    the connection. No hostile nameserver is needed -- the stub stands in for one with TTL 0.

    What is asserted is the pairing rather than the mechanism: the value the request is opened with is
    the address the guard approved, and the name is resolved once, not once per consumer of it."""
    answers = iter([("93.184.216.34", 80), ("127.0.0.1", 80)])
    resolutions = []

    def fake_getaddrinfo(host, port, *a, **k):
        resolutions.append(host)
        return [(2, 1, 6, "", next(answers))]

    seen = []

    class Opener:
        def open(self, req, timeout=None):
            seen.append(web._PIN.ip)                     # what the connection would be told to dial
            return _Resp(b"<html>ok</html>", "text/html")

    monkeypatch.setattr(web.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(web, "_OPENER", Opener())
    web.Net(allow_private=False)._get("http://rebind.attacker.tld/")
    assert seen == ["93.184.216.34"], "the request went to the address that passed the check"
    assert len(resolutions) == 1, "one resolve per hop: a second resolve is a second answer"
    assert web._PIN.ip == "", "the pin is released: one left set would be applied to the next request"


def test_a_redirect_repins_to_its_own_address(monkeypatch):
    """Each hop is its own target. A pin set once for the first URL would carry the first host's address
    into the second request, so a redirect to a *different public* host would dial the wrong machine --
    which no amount of re-guarding the URL would catch, because the URL it checks would be correct."""
    answers = {"a.example": ("93.184.216.34", 80), "b.example": ("93.184.216.99", 80)}
    seen = []

    class HopError(urllib.error.HTTPError):
        def __init__(self, url):
            super().__init__(url, 302, "found", {"Location": "http://b.example/"}, None)

    class RedirectOpener:
        def __init__(self):
            self.n = 0

        def open(self, req, timeout=None):
            self.n += 1
            seen.append(web._PIN.ip)
            if self.n == 1:
                raise HopError(req.full_url)
            return _Resp(b"<html>end</html>", "text/html")

    monkeypatch.setattr(web.socket, "getaddrinfo",
                        lambda host, port, *a, **k: [(2, 1, 6, "", answers[host])])
    monkeypatch.setattr(web, "_OPENER", RedirectOpener())
    _body, _ctype, final = web.Net(allow_private=False)._get("http://a.example/")
    assert final == "http://b.example/"
    assert seen == ["93.184.216.34", "93.184.216.99"], "the second hop is checked and dialled as itself"


def test_an_environment_proxy_is_not_used(monkeypatch):
    """A proxy resolves the name on its own, which is exactly the second opinion this fix removes, so the
    guard and a silent proxy cannot both be honoured and the guard wins.

    Asserted as dispatch, not as the presence of a handler: `OpenerDirector.add_handler` only keeps a
    handler that contributes a `<protocol>_open` method, so an empty `ProxyHandler` is dropped rather than
    installed -- an assertion that it is present and empty could never hold. With the environment pointing
    at a proxy, what matters is that nothing in the http/https dispatch chain can route through one, and
    that the pinned handler is there instead.
    """
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9999")
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9999")
    # Built here, inside the environment being described. An opener constructed at import would already
    # have read a proxy-free environment, and the assertion would pass against code that proxies. The
    # first version of this test did exactly that and survived a mutant that re-allowed the proxy.
    for opener, whence in ((web._build_opener(), "a fresh opener"), (web._OPENER, "the shipped opener")):
        for scheme in ("http", "https"):
            chain = opener.handle_open.get(scheme, [])
            assert chain, f"{whence}: nothing would handle {scheme}"
            assert not any(isinstance(h, urllib.request.ProxyHandler) for h in chain), \
                f"{whence}: {scheme} would be routed through a proxy the guard never vetted"
            assert any(isinstance(h, (web._PinnedHTTPHandler, web._PinnedHTTPSHandler))
                       for h in chain), f"{whence}: {scheme} is not handled by the pinning handler"


def test_fetch_reports_kept_of_fetched_and_labels_untrusted(monkeypatch):
    page = b"<html><head><style>x{}</style></head><body><h1>Hello</h1><p>Body text here.</p></body></html>"
    net = web.Net()
    monkeypatch.setattr(net, "_get", lambda url, **k: (page, "text/html; charset=utf-8", url))
    report = net.fetch("http://8.8.8.8/")
    assert "fetched http://8.8.8.8" in report
    assert "kept" in report and "of" in report and "UTC" in report        # declared loss + timestamp
    assert "untrusted web page" in report                                  # injection labelling
    assert "Hello" in report and "Body text here." in report
    assert "x{}" not in report                                             # <style> content dropped


def test_fetch_notes_a_redirect(monkeypatch):
    net = web.Net()
    monkeypatch.setattr(net, "_get", lambda url, **k: (b"hi", "text/plain", "http://elsewhere/final"))
    assert "redirected from" in net.fetch("http://8.8.8.8/start")


def test_fetch_turns_a_transport_error_into_text(monkeypatch):
    net = web.Net()

    def boom(url, **k):
        raise urllib.error.URLError("no route to host")
    monkeypatch.setattr(net, "_get", boom)
    assert "could not reach" in net.fetch("http://8.8.8.8/")


# --- the redirect re-check inside the real _get -------------------------------------------------

def test_redirect_into_the_lan_is_refused(monkeypatch):
    """The oldest SSRF trick: a public URL that 302s to 127.0.0.1. `fetch` disabled urllib's auto-
    redirect precisely so this hop is re-guarded. The public hop is allowed and answers with a
    redirect; the private hop must be refused *before* it is ever opened. Drop the re-guard and the
    second hop is opened and served -- so this fails on the real thing, not a StopIteration."""
    class Opener:
        def open(self, req, timeout=None):
            url = req.full_url
            if "127.0.0.1" in url:
                return _Resp(b"SECRET-ADMIN", "text/plain")   # would leak if reached
            raise urllib.error.HTTPError(url, 302, "Found",
                                         {"Location": "http://127.0.0.1/admin"}, None)
    monkeypatch.setattr(web, "_OPENER", Opener())
    with pytest.raises(web._Blocked):
        web.Net(allow_private=False)._get("http://8.8.8.8/")   # public target allowed, then refused


# --- search: the operator endpoint is exempt from the guard; parsing is backend-agnostic --------

_SEARX = ('{"results": [{"url": "https://a.example/x", "title": "Alpha", '
          '"content": "the alpha <b>snippet</b>", "engine": "wikipedia"},'
          '{"url": "https://b.example", "title": "Beta", "content": "beta", "engine": "brave"}]}')

_DDG = ('<div><a class="result__a" href="//duckduckgo.com/l/?uddg='
        'https%3A%2F%2Fddg.example%2Fpage&amp;rut=abc">DDG Result</a>'
        '<a class="result__snippet">a duck snippet</a></div>')


def test_search_works_against_a_localhost_endpoint_though_fetch_would_not(monkeypatch):
    """The search endpoint is the operator's, not the model's, so it is exempt from the address guard
    — which is exactly what lets the default backend be a localhost SearXNG. If search ever routed
    through the guard, this default host would be refused and the feature would be dead out of the box."""
    net = web.Net(allow_private=False, search_url="http://127.0.0.1:8888/search?q=%s&format=json")
    # Real `_get`, only the socket stubbed: if search ever routed through the guard, this localhost
    # endpoint would raise _Blocked (which search does not catch) and the test would error red.
    monkeypatch.setattr(web, "_OPENER", _FixedOpener(_SEARX.encode(), "application/json"))
    out = net.search("anything")
    assert "Alpha" in out and "https://a.example/x" in out
    assert "engine: wikipedia" in out              # the whose-ranking attribution


def test_search_reports_an_unreachable_endpoint_by_name(monkeypatch):
    net = web.Net(search_url="http://127.0.0.1:9/search?q=%s")

    def boom(url, **k):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(net, "_get", boom)
    out = net.search("q")
    assert "could not reach the search endpoint" in out and "connection refused" in out


def test_search_never_prose_without_a_query():
    assert "non-empty query" in web.Net().search("   ")


# --- parsers and decoders -----------------------------------------------------------------------

def test_searx_json_parses_with_per_result_engine():
    results = web._search_results(_SEARX)
    assert len(results) == 2
    assert results[0]["url"] == "https://a.example/x"
    assert results[0]["provider"] == "wikipedia"
    assert results[0]["snippet"] == "the alpha snippet"      # <b> markup stripped


def test_ddg_html_parses_and_unwraps_the_redirect_wrapper():
    results = web._search_results(_DDG)
    assert len(results) == 1
    assert results[0]["url"] == "https://ddg.example/page"   # uddg decoded, entity unescaped
    assert results[0]["title"] == "DDG Result"
    assert results[0]["snippet"] == "a duck snippet"


def test_gzip_body_is_inflated_not_mojibake():
    # A server may gzip despite Accept-Encoding: identity; left raw it decodes to junk that reads as a
    # broken fetch, so the body is inflated before it is decoded rather than after.
    assert web._decompress(gzip.compress(b"hello world")) == b"hello world"
    assert web._decompress(b"plain") == b"plain"


def test_a_gzip_bomb_inflates_to_at_most_the_fetch_ceiling(monkeypatch):
    # The ceiling is on the *compressed* read, so an inflated body is the one place a page can be
    # larger than everything the run agreed to hold. Deflate reaches ~1000:1 on constant bytes, so a
    # 2 MB ceiling on the wire is a gigabyte in the process. The cap must be the module's own, read
    # at call time: a default bound at def time would keep the old number when the cap is retuned,
    # and a run would hold more than its ceiling while every test still passed.
    monkeypatch.setattr(web, "MAX_FETCH_BYTES", 100_000)
    bomb = gzip.compress(b"\0" * 8_000_000)          # ~8 KB on the wire, 8 MB in the model's mouth
    assert len(bomb) < 100_000, "the bomb must fit under the ceiling while compressed"
    out = web._decompress(bomb)                       # the production call shape: no limit argument
    assert len(out) == 100_000, f"inflated to {len(out)} bytes against a 100_000 ceiling"
    assert out == b"\0" * 100_000, "what it kept must be the front of the page, not padding"


class _SpyZlib:
    """A stand-in for the zlib module. `web.py` reaches the decoder through its own name, so a test can
    count what the decoder produced without mutating the stdlib for the rest of the process."""

    MAX_WBITS = zlib.MAX_WBITS
    error = zlib.error

    def __init__(self):
        self.produced = 0

    def decompressobj(self, *args, **kwargs):
        spy = self
        real = zlib.decompressobj(*args, **kwargs)

        class Watched:
            def decompress(self, data, max_length=-1):
                out = real.decompress(data, max_length)
                spy.produced += len(out)
                return out

            def flush(self, *length):
                out = real.flush(*length)
                spy.produced += len(out)
                return out

            def __getattr__(self, name):
                return getattr(real, name)

        return Watched()


def test_the_inflate_never_produces_more_than_the_ceiling(monkeypatch):
    # The ceiling is a promise about *memory*, and the return value is not where memory shows up. A
    # final slice bounds the result and nothing else: measured, an 8 MB body capped at 100 KB still
    # expanded the full 8 MB behind the slice, and a `flush()` call handed back 7.9 MB of it in one go.
    # So the guarantee is asserted on the decoder — every byte it was asked to make — not on the length
    # of what the function hands back, which two mutants proved could agree while the run died.
    spy = _SpyZlib()
    monkeypatch.setattr(web, "zlib", spy)
    out = web._decompress(gzip.compress(b"\0" * 8_000_000), 100_000)
    assert len(out) == 100_000
    assert spy.produced == 100_000, (f"the decoder produced {spy.produced} bytes against a 100_000 "
                                     "ceiling: the result was trimmed, the memory was not")


def test_a_stream_that_errors_comes_back_as_the_bytes_that_arrived():
    # Two shapes of broken, and they are not the same. A stream that merely *ends* early (what a read
    # capped at the ceiling leaves) yields its prefix — see the cut-body test. A stream that *errors*
    # yields nothing: measured, `decompress` raises before it returns, so whatever had inflated is
    # discarded inside the decoder and no `except` clause can recover it. Salvaging it would mean
    # feeding the body in slices so earlier slices' output survives in the caller; that machinery is
    # bought only for corrupt input, while the capped-read case is the one a real fetch meets. So the
    # bytes come back unchanged and the decode step says what it found — an empty string here would
    # read as an empty page, which is a smoother lie than the one being refused.
    mid = bytearray(gzip.compress(b"z" * 20_000))
    mid[len(mid) // 2] ^= 0xFF
    assert web._decompress(bytes(mid), 1_000_000) == bytes(mid)
    header = b"\x1f\x8b\xff\x00not really a deflate stream at all"
    assert web._decompress(header, 1_000_000) == header


def test_concatenated_members_past_the_ceiling_stop_at_the_ceiling(monkeypatch):
    # The ceiling lives in the loop condition, not only inside one member's inflate. A finished member
    # with another behind it hands the loop fresh bytes, and without that guard the next call is asked
    # for `cap - len(out)` — zero, or negative — which is where a bound silently stops being a bound.
    spy = _SpyZlib()
    monkeypatch.setattr(web, "zlib", spy)
    whole = gzip.compress(b"a" * 150_000) + gzip.compress(b"b" * 150_000)
    out = web._decompress(whole, 200_000)
    assert len(out) == 200_000 and spy.produced == 200_000, \
        f"{len(out)} returned, {spy.produced} produced, against a 200_000 ceiling"
    assert out == (b"a" * 150_000 + b"b" * 50_000), "what it kept must be the front of the page"


def test_the_fetch_ceiling_reaches_the_inflate_of_a_real_response(monkeypatch):
    # The ceiling is a promise about one request, so the request's own number has to travel to the
    # inflate. A decoder left on the module default would honour a caller who passed a smaller cap
    # for one page while inflating ten times that — and every unit test of `_decompress` alone,
    # including the ceiling tests, would stay green while it did.
    monkeypatch.setattr(web, "_OPENER", _FixedOpener(gzip.compress(b"\0" * 8_000_000), "text/html"))
    body, _ctype, _url = web.Net()._get("https://example.invalid/p", guard=False, max_bytes=200_000)
    assert len(body) == 200_000, f"inflated to {len(body)} against a 200_000 request cap"


def test_a_gzip_stream_cut_mid_body_yields_the_prefix_it_inflates():
    # This is not an exotic input: reading a page with a cap on the wire *is* a truncated stream, so
    # the common large-page fetch arrives here broken on purpose. Returning the raw bytes for it — the
    # shape an exception-based decoder falls back to — hands the decode step a compressed body, which
    # is mojibake that reads as a broken fetch. The prefix that did inflate is real text, and a capped
    # fetch already has a disclosure line for being capped.
    whole = gzip.compress(b"<html>" + b"y" * 200_000 + b"</html>")
    cut = whole[: len(whole) // 2]
    out = web._decompress(cut, 1_000_000)
    assert out.startswith(b"<html>"), f"got {out[:24]!r}: the raw body came back, not the text"
    assert out != cut, "a truncated gzip body must not be returned as it arrived"


def test_concatenated_gzip_members_both_reach_the_reader():
    # Legal and seen in the wild (a log rotated mid-write, a proxy that appends). gzip.decompress
    # handled every member; a decoder that stopped at the first eof would silently drop the rest of
    # the page while reporting a complete fetch, which is the failure this module names elsewhere.
    whole = gzip.compress(b"first ") + gzip.compress(b"second")
    assert web._decompress(whole, 1_000_000) == b"first second"


def test_a_body_under_the_ceiling_is_not_cut_by_the_ceiling(monkeypatch):
    # The cap exists to stop a bomb, not to truncate an ordinary page. A limit applied eagerly (or a
    # decoder that stops at the first block) would silently shorten the small fetches that work today.
    monkeypatch.setattr(web, "MAX_FETCH_BYTES", 100_000)
    whole = gzip.compress(b"<html>" + b"x" * 5_000 + b"</html>")
    assert web._decompress(whole).startswith(b"<html>")
    assert web._decompress(whole).endswith(b"</html>")


def test_decode_reduces_html_but_passes_json_and_describes_binary():
    text, kind = web._decode(b"<p>Just  text</p>", "text/html")
    assert text == "Just  text" or text == "Just text"
    assert kind == "text"
    jtext, jkind = web._decode(b'{"a": 1}', "application/json")
    assert jtext == '{"a": 1}' and "json" in jkind
    btext, bkind = web._decode(b"\x00\x01\x02binary", "application/octet-stream")
    assert bkind == "binary" and "not shown" in btext


# --- the extension only offers the tools; wiring/gating is the CLI's job ------------------------

def test_extension_offers_both_tools_bound_to_the_net():
    ext = web.extension(web.Net())
    names = sorted(t.__name__ for t in ext.tools)
    assert names == ["web_fetch", "web_search"]


# --- helper (none needed beyond _Resp/_FixedOpener above) -------------------------------------


def test_empty_search_names_a_throttled_engine_rather_than_claiming_nothing():
    """An empty result set whose upstream engines were suspended must say so. 'found nothing' when the
    engines were rate-limited reads as 'the web has nothing about X' and the model stops looking."""
    body = ('{"query": "x", "results": [], "unresponsive_engines": '
            '[["brave", "Suspended: too many requests"]]}')
    net = web.Net(search_url="http://127.0.0.1:8888/search?q=%s&format=json")
    net._get = lambda url, **k: (body.encode(), "application/json", url)
    out = net.search("x")
    assert "no results" in out and "brave" in out and "too many requests" in out
    assert "found nothing" not in out


def test_a_plain_empty_search_says_the_query_matched_nothing():
    body = '{"query": "x", "results": [], "unresponsive_engines": []}'
    net = web.Net(search_url="http://127.0.0.1:8888/search?q=%s")
    net._get = lambda url, **k: (body.encode(), "application/json", url)
    assert "matched nothing" in net.search("x")


def test_a_large_fetch_hands_over_the_whole_page_so_the_rest_is_still_reachable(monkeypatch):
    """This test asserted the opposite, and the reversal is the finding.

    `fetch` clipped the page at the producer limit, so the rest of the page existed nowhere: not in the
    context, not in a file, not reachable by any tool the run held. The disclosure was honest about the
    loss ("showing the first ~N lines") and useless about recovering it. Clipping below the result limit
    bought one thing -- no copy of a stranger's page under the run's `out/` -- and cost the page.

    A page is ephemeral in a way a file is not: the run cannot re-open a URL and get the same bytes, so
    the producer rule (stop at BODY limits so your own disclosure survives) trades the wrong way here.
    Handing the whole thing to the choke point costs a file the run owns and buys a pointer it can follow.
    """
    page = ("<html><body>" + "".join(f"<p>line {i} of the page</p>" for i in range(6000))
            + "<p>THE LAST PARAGRAPH</p></body></html>").encode("utf-8")
    net = web.Net()
    monkeypatch.setattr(net, "_get", lambda url, **k: (page, "text/html; charset=utf-8", url))
    report = net.fetch("http://8.8.8.8/")

    assert len(report.encode("utf-8")) > ingest.MAX_BYTES, "the page was pre-clipped: the rest is gone"
    assert "THE LAST PARAGRAPH" in report, "the end of the page did not survive the fetch"
    assert "source was larger" not in report, "a disclosure that the rest is unreachable is a lie now"
    assert "untrusted web page" in report and "kept" in report


def test_a_fetch_that_hit_its_own_ceiling_says_so_on_the_decoded_text(monkeypatch):
    """The one loss that is real. `_get` stops at MAX_FETCH_BYTES, so past that the page genuinely was
    never read and no spill recovers it. That is disclosed, where the recoverable loss above is not."""
    page = b"a" * web.MAX_FETCH_BYTES
    net = web.Net()
    monkeypatch.setattr(net, "_get", lambda url, **k: (page, "text/plain", url))
    report = net.fetch("http://8.8.8.8/")
    assert "fetch limit" in report and "never read" in report
    assert len(report.encode("utf-8")) >= web.MAX_FETCH_BYTES



# --- WEB-4: the protocol is not the operator's call to make --------------------------------------

def test_the_scheme_is_refused_even_on_the_paths_that_skip_the_address_guard():
    """`guard=False` has always meant "the operator chose this address, do not vet the address". It has
    never meant "any protocol", but that is what it became: `build_opener` adds `FileHandler`,
    `DataHandler` and `FTPHandler` whether or not they are asked for, so a search URL written as
    `file:///…` opened a local file and handed its bytes to the caller, and `data:` was readable too.

    The address being trusted is a decision the operator made by setting `SAH_SEARCH_URL`. Reading the
    disk is not a decision anybody wants to make twice, by accident."""
    net = web.Net()
    for url in ("file:///etc/passwd", "data:text/plain,payload", "ftp://example.invalid/x"):
        try:
            body, ctype, _ = net._get(url, guard=False)
        except web._Blocked:
            continue
        raise AssertionError(f"{url} was opened and returned {len(body)} bytes as {ctype!r}")


def test_a_search_endpoint_that_bounces_to_a_file_is_refused(monkeypatch):
    """The redirect loop re-checks the address at every hop, which is the whole point of pinning. With
    the address check switched off for the operator's endpoint, the *hop* itself was unchecked, so a
    single 302 to `file:///` turned a search into a local read — the misconfiguration that needs no
    attacker, only a `Location` header."""
    class HopError(urllib.error.HTTPError):
        def __init__(self, url):
            super().__init__(url, 302, "found", {"Location": "file:///etc/passwd"}, None)

    class BouncyOpener:
        def __init__(self):
            self.n = 0

        def open(self, req, timeout=None):
            self.n += 1
            if self.n == 1:
                raise HopError(req.full_url)
            return _Resp(b"root:x:0:0:0:0:0:0", "text/plain")   # reached only if the hop went unchecked

    monkeypatch.setattr(web, "_OPENER", BouncyOpener())
    try:
        web.Net()._get("http://search.example/search?q=x", guard=False)
    except web._Blocked:
        return
    raise AssertionError("a search endpoint bounced the request to file:// and it was opened")


def test_http_is_still_fine_when_the_scheme_check_runs(monkeypatch):
    """The control that keeps the two tests above a guard rather than a ban."""
    class OkOpener:
        def open(self, req, timeout=None):
            return _Resp(b"<html>fine</html>", "text/html")
    monkeypatch.setattr(web, "_OPENER", OkOpener())
    body, ctype, _ = web.Net()._get("http://example.invalid/", guard=False)
    assert b"fine" in body


# --- WEB-6: a stranger's JSON shape is not a crash ------------------------------------------------

def _stub_search_get(monkeypatch, net, body: bytes, ctype: str = "application/json") -> None:
    monkeypatch.setattr(net, "_get", lambda url, **k: (body, ctype, url))


def test_a_search_endpoint_answering_a_json_list_names_the_shape(monkeypatch):
    """A body that is valid JSON and the wrong shape is the most common way to mis-point
    `SAH_SEARCH_URL` — an API that returns a bare array, a proxy returning an error document — and it
    reached `.get(...)` on a list. The loop turns the exception into a failed call, so the run learned
    only that something broke; what it needs is that the endpoint answered in a shape it cannot read,
    which is the sentence that says to go look at the URL."""
    net = web.Net(search_url="http://127.0.0.1:9/search?q=%s")
    _stub_search_get(monkeypatch, net, b'[{"title": "x", "url": "http://a"}]')
    out = net.search("q")
    assert "shape" in out, f"a wrong-shaped answer was not named: {out[:160]!r}"
    assert "matched nothing" not in out, "a parse failure was reported as an empty web"


def test_malformed_engine_diagnostics_do_not_become_a_crash(monkeypatch):
    """`unresponsive_engines` is documented as a list of pairs, and SearXNG's own field has been a dict
    and has carried three-element rows. Unpacking either raised, and the raise escaped before the code
    that names why a result set is empty could run — so a throttled upstream became a crash, and a crash
    says less than the lie it replaced.

    What is asserted is that the outage is still *named*, not merely that nothing threw: a mutant that
    drops the dict branch returns an empty reason and the old test passed against it, reporting "matched
    nothing" about an upstream that said "google: timeout"."""
    for body in (b'{"results": [], "unresponsive_engines": {"google": "timeout"}}',
                 b'{"results": [], "unresponsive_engines": [["google", "timeout", "extra"]]}',
                 b'{"results": [], "unresponsive_engines": [["google", "captcha"]]}'):
        net = web.Net(search_url="http://127.0.0.1:9/search?q=%s")
        _stub_search_get(monkeypatch, net, body)
        out = net.search("q")
        assert "google" in out and "timeout" in out.replace("captcha", "timeout"), \
            f"the outage was not named for {body[:56]!r}: {out[:150]!r}"


def test_a_response_whose_results_field_is_not_a_list_says_so(monkeypatch):
    """A `results` field that parses, exists, and is unusable. Reporting it as "matched nothing" tells
    the model the web is empty; the fault is one field deep in the endpoint's answer, and only the second
    sentence is worth acting on.

    The bare number is the case that says whether the parser or only the disclosure is defending here: a
    number is not iterable, so a loop over it raises inside the parser rather than returning nothing."""
    for body in (b'{"results": "not a list"}', b'{"results": {"0": "x"}}', b'{"results": 5}'):
        net = web.Net(search_url="http://127.0.0.1:9/search?q=%s")
        _stub_search_get(monkeypatch, net, body)
        out = net.search("q")
        assert "shape" in out, f"an unusable results field was reported as an empty web: {out[:150]!r}"
        assert "matched nothing" not in out


def test_a_response_with_no_results_field_at_all_names_the_field(monkeypatch):
    """A JSON object that simply is not a search response — a proxy's error document, an API root, a
    health endpoint — is the most likely way a configured URL turns out to be wrong, so it is the case
    most worth getting right. It used to raise a KeyError on the way to explaining itself."""
    for body in (b'{}', b'{"error": "upstream failed", "status": 502}', b'{"status": "ok"}'):
        net = web.Net(search_url="http://127.0.0.1:9/search?q=%s")
        _stub_search_get(monkeypatch, net, body)
        out = net.search("q")
        assert "'results'" in out or "shape" in out, f"a non-response was not named: {out[:150]!r}"
        assert "matched nothing" not in out


def test_one_malformed_result_row_does_not_lose_the_readable_ones(monkeypatch):
    """The promise is per row: a stranger's API that inserts a string between two objects, or a number
    where a row should be, costs one result and not the whole page. Asserted by title, because "did not
    crash" was already covered — and passed against the mutant that dropped the row check."""
    body = (b'{"results": [{"title": "Good", "url": "http://good", "content": "fine"},'
            b' "a string between rows", 42, {"title": "Also good", "url": "http://also"}]}')
    net = web.Net(search_url="http://127.0.0.1:9/search?q=%s")
    _stub_search_get(monkeypatch, net, body)
    out = net.search("q")
    assert "Good" in out and "Also good" in out, f"readable rows were lost: {out[:200]!r}"
    assert "2 result(s)" in out


# --- WEB-7: a page that says what it is encoded as should be read as that -------------------------

def test_a_declared_charset_is_the_one_used():
    """`charset=` in the Content-Type is the server telling us how to read its bytes. Decoding them as
    UTF-8 anyway turns `café` into `caf` plus a replacement character — and a replacement character
    *reads like text*, which is the failure mode this repo has already ruled on for bytes it does not
    choose."""
    text, _kind = web._decode("café".encode("iso-8859-1"), "text/html; charset=iso-8859-1")
    assert "café" in text, f"the declared charset was ignored: {text[:60]!r}"


def test_html_that_declares_its_charset_only_in_a_meta_is_decoded_by_it():
    """Plenty of pages send `Content-Type: text/html` with no charset and put the declaration in the
    document, where the standard says a reader should find it. Those pages are not broken; the reader
    was."""
    body = ('<html><head><meta charset="shift_jis"></head><body>日本語</body></html>'
            .encode("shift_jis"))
    text, _kind = web._decode(body, "text/html")
    assert "日本語" in text, f"the in-document declaration was ignored: {text[:60]!r}"


def test_an_unknown_charset_falls_back_rather_than_failing():
    """`charset=banana` is a real thing servers send. A LookupError out of the decoder would fail a fetch
    whose bytes are perfectly readable."""
    text, _kind = web._decode(b"still readable", "text/plain; charset=definitely-not-an-encoding")
    assert "still readable" in text


def test_a_search_response_is_decoded_by_its_charset_too(monkeypatch):
    """The search path decoded UTF-8 on its own, twice, rather than going through the one decoder — so a
    latin-1 endpoint produced titles full of replacement characters, and the model read them as the
    titles."""
    body = ('{"results": [{"title": "café", "url": "http://x", "content": "résumé"}]}'
            .encode("iso-8859-1"))
    net = web.Net(search_url="http://127.0.0.1:9/search?q=%s")
    # The charset has to be declared for this to be a test of honouring it. With no declaration, UTF-8 is
    # the right guess and the bytes are genuinely damaged, so asserting on that body would ask the
    # decoder to invent an encoding from nothing.
    _stub_search_get(monkeypatch, net, body, "application/json; charset=iso-8859-1")
    out = net.search("q")
    assert "café" in out, f"the endpoint's charset was ignored: {out[:200]!r}"
