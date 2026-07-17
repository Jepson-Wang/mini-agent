"""[M1] web_fetch：抓取一个网页，返回正文文本。

这是第一个「危险」工具。read_file 的参数好歹还在本机文件系统里，而 web_fetch 的
url 是**模型说了算**的——它可以是 http://localhost:8080/admin，也可以是云主机的
元数据地址 http://169.254.169.254/latest/meta-data/（AWS / 阿里云都在这个地址上），
一把梭就是标准的 SSRF。所以这个文件里超过一半的代码在做「拒绝」，而不是在做「抓取」。

四道闸：
  1. scheme 白名单     —— 只放行 http/https，挡掉 file:// ftp:// data:// 等
  2. 目标 IP 检查      —— 解析域名后逐个 IP 校验，挡掉内网/回环/链路本地/保留地址
  3. 重定向逐跳复检    —— 公网 URL 302 到 127.0.0.1 是最经典的绕过手法
  4. 响应体大小上限    —— 不信 Content-Length，边读边数，防止一个大文件撑爆内存

另外做了 HTML 正文抽取：把原始 HTML 直接塞回模型是在烧 context——一个普通页面的
标签和内联 script/style 能占掉八成 token，对模型理解内容却毫无帮助。

只用标准库（urllib + html.parser + ipaddress），不引入 requests / beautifulsoup。
"""
from __future__ import annotations

import ipaddress
import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from mini_agent.log import get_logger
from mini_agent.tools.registry import registry

logger = get_logger(__name__)

# 返回给模型的字符上限：8000 字符 ≈ 2000+ token，够理解一个页面，又不至于挤爆上下文
DEFAULT_MAX_CHARS = 8000
# 响应体硬上限：先按字节掐断，防止「一个 500MB 的 iso」把内存吃光
MAX_BYTES = 2 * 1024 * 1024
TIMEOUT_SECONDS = 10
USER_AGENT = "mini-agent/0.1 (learning project)"

ALLOWED_SCHEMES = {"http", "https"}


class WebToolError(Exception):
    """策略拒绝（scheme 不合法、目标是内网……）。会被转成 {"error": ...} 回给模型。"""


# ---------------------------------------------------------------------------
# 闸 1 + 闸 2：URL 与目标 IP 校验
# ---------------------------------------------------------------------------

def _check_url(url: str) -> None:
    """URL 不合规就抛 WebToolError；合规则静默返回。"""
    parts = urllib.parse.urlparse(url)

    if parts.scheme not in ALLOWED_SCHEMES:
        # file:// 能读本机任意文件，data:// 能绕过一切网络检查——直接堵死
        raise WebToolError(f"scheme 不被允许: {parts.scheme!r}（只放行 http / https）")
    if not parts.hostname:
        raise WebToolError(f"URL 缺少 host: {url!r}")

    _check_host_is_public(parts.hostname)


def _check_host_is_public(hostname: str) -> None:
    """解析域名，确认它指向的**每一个** IP 都在公网上。

    为什么要逐个查而不是只查第一个：一个域名可以解析出多条 A 记录，只校验第一条
    的话，让第二条指向 127.0.0.1 就绕过去了。

    已知残留风险（MVP 接受）：DNS rebinding。我们校验的是这一次解析的结果，真正
    建连时 urllib 会**再解析一次**，中间这个窗口里 DNS 可以改答案。要彻底堵死得
    自己 pin 住 IP 去连、并手工带上 Host 头，成本远超 MVP 的收益。
    """
    try:
        # 只要地址不要端口；不限定 family，IPv4 / IPv6 都查
        infos = socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise WebToolError(f"域名解析失败: {hostname} ({e})") from e

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private          # 10.x / 172.16.x / 192.168.x —— 内网
            or ip.is_loopback      # 127.0.0.1 —— 本机上的服务
            or ip.is_link_local    # 169.254.x —— 云元数据地址就在这儿
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified   # 0.0.0.0
        ):
            raise WebToolError(
                f"拒绝访问非公网地址: {hostname} -> {ip}。"
                "web_fetch 只能抓公网页面，不能用来探测内网或本机服务。"
            )


# ---------------------------------------------------------------------------
# 闸 3：重定向逐跳复检
# ---------------------------------------------------------------------------

class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """每一跳重定向都重新过一遍 URL 检查。

    只查最初那个 URL 是不够的：一个完全正常的公网 URL，可以 302 到
    http://169.254.169.254/。这是 SSRF 最经典的绕过手法，也是真实 CVE 的常客。
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_url(newurl)   # 不合规就在这里抛，连都不去连
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_GuardedRedirectHandler)


# ---------------------------------------------------------------------------
# HTML → 正文
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """把 HTML 抽成纯文本：丢掉标签，并整段跳过 script / style 里的内容。

    不追求完美（那是 beautifulsoup 的活），只要把 token 从「八成是尖括号」降到
    「基本是人话」就达到目的了。
    """

    _SKIP = {"script", "style", "noscript", "template"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return "\n".join(self._chunks)


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:  # 畸形 HTML 不该让整个工具失败，能抽多少算多少
        logger.warning("HTML 解析中断，返回已抽取的部分")
    return parser.text()


def _looks_textual(content_type: str) -> bool:
    ct = content_type.lower()
    if not ct:
        return True   # 服务器没给 Content-Type 时，按文本试一把
    return ct.startswith("text/") or any(
        k in ct for k in ("json", "xml", "html", "javascript", "csv")
    )


def _charset_of(content_type: str) -> str:
    for part in content_type.split(";"):
        part = part.strip().lower()
        if part.startswith("charset="):
            return part.split("=", 1)[1].strip("\"'") or "utf-8"
    return "utf-8"


def _web_enabled() -> bool:
    """check_fn：没开 ALLOW_WEB 就不把这个工具暴露给模型。

    默认关闭是有意的——一个能上网的 agent 和一个不能上网的 agent，风险面完全
    不是一个量级。要用就显式开（.env 里 ALLOW_WEB=1）。
    """
    return os.getenv("ALLOW_WEB", "").lower() in {"1", "true", "yes"}


# ---------------------------------------------------------------------------
# handler
# ---------------------------------------------------------------------------

def _err(message: str) -> str:
    """web 工具的错误出口：永远是 {"error": ...} 的合法 JSON 字符串。"""
    return json.dumps({"error": message}, ensure_ascii=False)


def _guarded_get(
    url: str, max_bytes: int, user_agent: str = USER_AGENT
) -> tuple[str, int, str, bytes, bool]:
    """过完四道闸做一次 GET，返回 (最终URL, 状态码, Content-Type, 原始字节, 是否截断)。

    web_fetch 和 web_search 共用这段：闸 1+2 在 _check_url，闸 3 在 _opener 的重定向
    复检，闸 4 是这里只读 max_bytes+1 字节。策略拒绝抛 WebToolError，网络错误抛
    urllib/socket 异常，都留给调用方转成 {"error": ...}。
    """
    _check_url(url)                                    # 闸 1 + 闸 2
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with _opener.open(req, timeout=TIMEOUT_SECONDS) as resp:   # 闸 3 在此生效
        status = resp.status
        final_url = resp.url                           # 重定向后的最终地址
        content_type = resp.headers.get("Content-Type", "")
        raw = resp.read(max_bytes + 1)                 # 闸 4：多读 1 字节判断截断
    truncated = len(raw) > max_bytes
    return final_url, status, content_type, raw[:max_bytes], truncated


def web_fetch(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """抓取 url，返回正文文本（JSON 字符串）。

    handler 的契约：**永远返回 JSON 字符串，绝不把异常抛给模型看**。可预期的失败
    （策略拒绝、HTTP 4xx/5xx、超时）都转成 {"error": ...}，带上足够模型判断
    「该重试还是换个 url」的信息。意料之外的异常冒出去，由 dispatch 兜底。
    """
    max_chars = max(1, min(int(max_chars), DEFAULT_MAX_CHARS * 4))

    try:
        final_url, status, content_type, raw, body_truncated = _guarded_get(url, MAX_BYTES)
    except WebToolError as e:                          # scheme / 内网 / 重定向被拒
        return _err(str(e))
    except urllib.error.HTTPError as e:
        return json.dumps({"error": f"HTTP {e.code} {e.reason}", "url": url}, ensure_ascii=False)
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        return json.dumps(
            {"error": f"请求失败: {type(e).__name__}: {e}", "url": url}, ensure_ascii=False
        )

    if not _looks_textual(content_type):
        # 二进制内容（图片 / PDF / 压缩包）没必要塞给模型，回一条元信息让它换路子
        return json.dumps(
            {"error": f"不是文本内容: Content-Type={content_type!r}，未读取正文",
             "url": final_url, "status": status},
            ensure_ascii=False,
        )

    text = raw.decode(_charset_of(content_type), errors="replace")
    if "html" in content_type.lower():
        text = _html_to_text(text)

    truncated = body_truncated or len(text) > max_chars
    text = text[:max_chars]

    return json.dumps(
        {
            "url": final_url,          # 最终地址：让模型知道自己被重定向了
            "status": status,
            "content_type": content_type,
            "truncated": truncated,    # 明确告诉模型「后面还有」，别把残篇当全文
            "chars": len(text),
            "text": text,
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# web_search：DuckDuckGo（免 key）
# ---------------------------------------------------------------------------
# DDG 的 html 端点 https://html.duckduckgo.com/html/?q=... 返回一页 HTML，
# 每条结果是 <a class="result__a" href="...">标题</a> + <a class="result__snippet">。
# href 常被包成 //duckduckgo.com/l/?uddg=<真实URL的百分号编码>，要解出来。
# 注意：这是非官方接口，会被限流、结构也可能变——学习够用，别当生产依赖。

# 用一个像浏览器的 UA，否则 DDG 常返回空页/挑战页而不是结果。
_SEARCH_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _decode_ddg_href(href: str) -> str:
    """把 DDG 的跳转链接 //duckduckgo.com/l/?uddg=<编码URL> 解回真实 URL。"""
    if href.startswith("//"):
        href = "https:" + href
    qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
    if "uddg" in qs:
        return qs["uddg"][0]
    return href


class _DuckDuckGoParser(HTMLParser):
    """从 DDG html 结果页抽出 [{title, url, snippet}, ...]。

    结果标题和摘要是两个相邻的 <a>，各带一个 class。命中 result__a 就新开一条记录、
    顺手解出真实 url；命中 result__snippet 就往最近那条上补摘要；到 </a> 收尾。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._mode: str | None = None      # "title" | "snippet" | None
        self._buf: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        a = dict(attrs)
        cls = a.get("class") or ""
        if "result__a" in cls:
            self.results.append(
                {"title": "", "url": _decode_ddg_href(a.get("href") or ""), "snippet": ""}
            )
            self._mode, self._buf = "title", []
        elif "result__snippet" in cls:
            self._mode, self._buf = "snippet", []

    def handle_data(self, data):
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self._mode and self.results:
            self.results[-1][self._mode] = "".join(self._buf).strip()
            self._mode, self._buf = None, []


def web_search(query: str, max_results: int = 5) -> str:
    """用 DuckDuckGo 搜索 query，返回排名靠前的结果（标题 + URL + 摘要）。

    只打 DDG 这一个固定的公网地址，模型无法把它指向内网——所以这个工具不像
    web_fetch 那样受 ALLOW_WEB 门控，开箱即用。
    """
    max_results = max(1, min(int(max_results), 10))
    ddg_url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query)

    try:
        _, _, content_type, raw, _ = _guarded_get(ddg_url, MAX_BYTES, _SEARCH_USER_AGENT)
    except WebToolError as e:
        return _err(str(e))
    except urllib.error.HTTPError as e:
        return _err(f"搜索失败: HTTP {e.code} {e.reason}")
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        return _err(f"搜索请求失败: {type(e).__name__}: {e}")

    parser = _DuckDuckGoParser()
    try:
        parser.feed(raw.decode(_charset_of(content_type), errors="replace"))
    except Exception:
        logger.warning("DuckDuckGo 结果解析中断，返回已抽取的部分")

    results = [r for r in parser.results if r["url"] and r["title"]][:max_results]
    if not results:
        return json.dumps(
            {"query": query, "count": 0, "results": [],
             "note": "没有解析到结果——可能被 DDG 限流，或页面结构变了"},
            ensure_ascii=False,
        )
    return json.dumps(
        {"query": query, "count": len(results), "results": results}, ensure_ascii=False
    )


# ---------------------------------------------------------------------------
# 自注册（必须在模块顶层——discover_builtin_tools 靠 AST 扫的就是这一句）
# ---------------------------------------------------------------------------

WEB_FETCH_SCHEMA = {
    "name": "web_fetch",
    "description": (
        "抓取一个公网网页并返回其正文文本（HTML 会被抽成纯文本）。"
        "只支持 http/https，不能访问内网、本机或云元数据地址。"
        "内容过长时会被截断，返回的 truncated 字段会标明。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "要抓取的完整 URL，必须以 http:// 或 https:// 开头",
            },
            "max_chars": {
                "type": "integer",
                "description": f"返回正文的最大字符数，默认 {DEFAULT_MAX_CHARS}",
            },
        },
        "required": ["url"],
    },
}

registry.register(
    name="web_fetch",
    toolset="web",
    schema=WEB_FETCH_SCHEMA,
    handler=web_fetch,
    check_fn=_web_enabled,
    is_async=False,
    description=WEB_FETCH_SCHEMA["description"],
)


WEB_SEARCH_SCHEMA = {
    "name": "web_search",
    "description": (
        "用 DuckDuckGo 搜索关键词，返回排名靠前的网页（标题、URL、摘要）。"
        "需要「查一下」「搜一下」「今天的新闻」这类信息时用它；"
        "拿到某个具体 URL 后再想读全文，用 web_fetch。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "返回结果条数，默认 5，最多 10",
            },
        },
        "required": ["query"],
    },
}

# 注意：web_search 不传 check_fn —— 它只打 DDG 一个固定公网地址，没有 SSRF 面，
# 所以不受 ALLOW_WEB 门控，开箱即用（web_fetch 能抓任意 URL，才需要门控）。
registry.register(
    name="web_search",
    toolset="web",
    schema=WEB_SEARCH_SCHEMA,
    handler=web_search,
    is_async=False,
    description=WEB_SEARCH_SCHEMA["description"],
)
