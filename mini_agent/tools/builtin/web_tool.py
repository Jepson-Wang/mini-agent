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

def web_fetch(url: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """抓取 url，返回正文文本（JSON 字符串）。

    handler 的契约：**永远返回 JSON 字符串，绝不把异常抛给模型看**。这里把可预期
    的失败（策略拒绝、HTTP 4xx/5xx、超时）都转成 {"error": ...}，并带上足够模型
    自己判断「该重试还是该换个 url」的信息。意料之外的异常就让它冒出去——
    registry.dispatch 会兜底包成 JSON（这就是两层包裹）。
    """
    max_chars = max(1, min(int(max_chars), DEFAULT_MAX_CHARS * 4))

    try:
        _check_url(url)                                    # 闸 1 + 闸 2
    except WebToolError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with _opener.open(req, timeout=TIMEOUT_SECONDS) as resp:   # 闸 3 在这里生效
            status = resp.status
            final_url = resp.url                           # 重定向后的最终地址
            content_type = resp.headers.get("Content-Type", "")
            # 闸 4：只读 MAX_BYTES+1，多出的那 1 字节用来判断「是不是被截断了」。
            # 不看 Content-Length——那是服务器说的，可以撒谎。
            raw = resp.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        return json.dumps(
            {"error": f"HTTP {e.code} {e.reason}", "url": url}, ensure_ascii=False
        )
    except WebToolError as e:                              # 重定向到了内网
        return json.dumps({"error": f"重定向被拒绝: {e}"}, ensure_ascii=False)
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        return json.dumps(
            {"error": f"请求失败: {type(e).__name__}: {e}", "url": url},
            ensure_ascii=False,
        )

    body_truncated = len(raw) > MAX_BYTES
    raw = raw[:MAX_BYTES]

    if not _looks_textual(content_type):
        # 二进制内容（图片 / PDF / 压缩包）没必要塞给模型，回一条元信息让它换路子
        return json.dumps(
            {
                "error": f"不是文本内容: Content-Type={content_type!r}，未读取正文",
                "url": final_url,
                "status": status,
            },
            ensure_ascii=False,
        )

    text = raw.decode(_charset_of(content_type), errors="replace")
    if "html" in content_type.lower():
        text = _html_to_text(text)

    truncated = body_truncated or len(text) > max_chars
    text = text[:max_chars]

    return json.dumps(
        {
            # 给最终地址而不是原始 url：模型该知道自己被重定向了
            "url": final_url,
            "status": status,
            "content_type": content_type,
            # 明确告诉模型「后面还有」，别把残篇当全文
            "truncated": truncated,
            "chars": len(text),
            "text": text,
        },
        ensure_ascii=False,
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
