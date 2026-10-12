"""Clean search-result text and URLs before they reach the model or the UI.

Search results are UNTRUSTED. Titles and snippets can contain HTML, scripts,
fake instructions or our own special tags. Here they are turned into plain
text that cannot look like a prompt format or an instruction.
"""
import html
import ipaddress
import re
import socket
from urllib.parse import urlsplit

_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?(</\1\s*>|$)", re.I | re.S)
_TAG = re.compile(r"<[^>]*>?", re.S)          # also removes <BOS>, <EOS>, <DOMAIN_KNOWLEDGE>, </x>, <|x|>
_SPECIAL_WORD = re.compile(r"\[(?:BOS|EOS|PAD|UNK)\]", re.I)
_CONTROL = re.compile("[\x00-\x1f\x7f-\x9f​-‏‪-‮⁦-⁩]")
_ROLE = re.compile(r"\b(user|assistant|question|answer|topic|system|human)\s*:", re.I)
_INSTRUCTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|your)\s+"
    r"(instructions?|prompts?|rules?|messages?)|you\s+are\s+now\b|new\s+instructions?\b|"
    r"(reveal|print|show)\s+(your\s+)?(system\s+)?(prompt|instructions)", re.I)
_SPACE = re.compile(r"\s+")
_BAD_URL_CHARS = re.compile("[\\s\x00-\x1f\x7f<>\"'`\\\\]")

ALLOWED_SCHEMES = ("http", "https")
MAX_URL_CHARS = 2048


def clean_text(value, max_chars=200):
    """Plain, single-line, length-limited text. Never raises."""
    text = "" if value is None else str(value)
    text = html.unescape(text)
    for _ in range(2):  # twice, in case unescaping revealed new tags (&lt;script&gt;)
        text = _SCRIPT_STYLE.sub(" ", text)
        text = _TAG.sub(" ", text)
        text = html.unescape(text)
    text = text.replace("<", " ").replace(">", " ")
    text = _CONTROL.sub(" ", text)
    text = _SPECIAL_WORD.sub(" ", text)
    text = _ROLE.sub(lambda m: m.group(1) + " -", text)   # "User:" -> "User -"
    text = _INSTRUCTION.sub("[removed]", text)
    text = _SPACE.sub(" ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-") + "..."
    return text


_INTERNAL_SUFFIXES = (".local", ".localhost", ".internal", ".lan", ".home", ".corp", ".intranet", ".localdomain")


def _public_ip(addr):
    if addr.version == 6 and addr.ipv4_mapped is not None:
        addr = addr.ipv4_mapped
    return bool(addr.is_global and not (addr.is_multicast or addr.is_loopback or addr.is_link_local
                                        or addr.is_private or addr.is_reserved or addr.is_unspecified))


def is_public_host(host):
    """False for localhost, internal-looking names and any IP that is not publicly routable
    (private, loopback, link-local incl. the 169.254.169.254 cloud-metadata address, reserved...).

    No DNS lookup is made: IP literals are checked directly (also the odd forms such as 2130706433,
    0x7f.1 and 127.1), names are checked by shape. Result pages are never fetched, so DNS
    rebinding is not a risk in this version; if fetching is added, resolve and re-check the IPs.
    """
    host = (host or "").lower().rstrip(".")
    if not host or host == "localhost" or host.endswith(_INTERNAL_SUFFIXES) or "." not in host and ":" not in host:
        return False
    try:
        return _public_ip(ipaddress.ip_address(host))
    except ValueError:
        pass
    last = host.rsplit(".", 1)[-1]
    if last.isdigit() or last.startswith("0x"):     # looks like a number, not a domain: legacy IPv4 forms
        try:
            return _public_ip(ipaddress.ip_address(socket.inet_aton(host)))
        except (OSError, ValueError):
            return False
    return True


def clean_url(value):
    """Return (url, domain) for a safe public http(s) URL, else None. Used for RESULT URLs only."""
    url = ("" if value is None else str(value)).strip()
    if not url or len(url) > MAX_URL_CHARS or _BAD_URL_CHARS.search(url):
        return None
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return None
    if parts.scheme.lower() not in ALLOWED_SCHEMES or not host or "@" in parts.netloc:
        return None
    if not re.fullmatch(r"[a-z0-9.-]+|[0-9a-f:]+", host.lower()):
        return None
    if not is_public_host(host):
        return None
    domain = host.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return url, domain
