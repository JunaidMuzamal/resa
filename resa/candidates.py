"""High-recall candidate generator for the recovery stage (the G1 contribution).

GLITCH only fires on parsed attributes/variables/comments, so it misses smells hidden inside shell
commands, connection strings, identifiers (``root_user``) and unparsed constructs. These relaxed,
line-level patterns deliberately over-generate; the recovery student and the verifier decide.
Stage-1 (GLITCH) warnings are removed before scoring, so recovery only ever *adds* detections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------- shared regexes
_VAR_REF = re.compile(r"(\{\{.*?\}\}|#\{.*?\}|\$\{.*?\}|\$[A-Za-z_]\w*|node\[|node\.\w|lookup\(|hiera\(|ENV\[|vault)", re.I)
_QUOTED = re.compile(r"""(["'])(?:(?!\1).)+\1""")
_COMMENT = {
    "ansible": re.compile(r"^\s*#|\s#\s|\{#"),
    "chef": re.compile(r"^\s*#|\s#\s|<%#"),
    "puppet": re.compile(r"^\s*#|\s#\s|^\s*//|/\*"),
}

_ADMIN_FLAG = re.compile(r"""(-u\s*['"]?|--user(?:name)?[=\s]+['"]?|-U\s*['"]?)(root|admin|administrator|sa|postgres)\b""", re.I)
_ADMIN_TOKEN = re.compile(r"(root|admin|administrator|superuser|sudo)", re.I)
_USER_CTX = re.compile(r"(user|login|owner|role|account|grant|become|uname|mysql|psql|mongo|connection|conn|credential|run_?as|sudo)", re.I)

_PWD_NAME = re.compile(r"(pass(word|wd|phrase)?|pwd|secret)", re.I)
_EMPTY_ASSIGN = re.compile(r"""(pass(word|wd)?|pwd|secret)\w*['"]?\s*(=>|=|:)\s*(""|''|nil\b|undef\b|$)""", re.I)
_EMPTY_FLAG = re.compile(r"""(-p\s*(""|'')|--password=(""|''|\s)|password\s*=\s*$)""", re.I)

_SECRET_NAME = re.compile(
    r"(pass(word|wd|phrase)?|pwd|secret|token|api[_-]?key|apikey|access[_-]?key|private[_-]?key|ssh[_-]?key|"
    r"credential|auth|user(name)?|login|cert|\bkey\b|salt|cookie)",
    re.I,
)
_ASSIGN = re.compile(r"""(=>|=|:)\s*(["']?)([^\s"',}#]{2,})""")
_URL_CRED = re.compile(r"://[^/\s:@]+:[^/\s@]+@")
_CMD_PWD = re.compile(r"""(-p\s*['"]?[^\s'"$#{-][^\s'"]*|--password[=\s]+['"]?[^\s'"$#{]+)""")

_INVALID_BIND = re.compile(r"(0\.0\.0\.0|['\"]::['\"]|\[::\]|INADDR_ANY)")
_BIND_CTX = re.compile(r"(bind|listen|host|address|addr|interface|ip)", re.I)

_SUSP_WORDS = re.compile(
    r"\b(bug|debug|todo|to-do|to_do|fixme|fix me|hack|xxx|workaround|temporar(y|ily)|insecure|unsafe|"
    r"not secure|security|disable[sd]?|bypass|kludge|later|should not|dont use|don't use|do not use|"
    r"hard-?coded|plaintext|plain text|unencrypted)\b",
    re.I,
)

_HTTP = re.compile(r"http://(?!(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]))", re.I)

_DOWNLOAD = re.compile(
    r"(get_url|wget|curl|remote_file|unarchive|archive|yum_repository|apt_repository|apt_key|zypper_repository|"
    r"rpm_key|download|source\s*=>\s*['\"]https?|url\s*[:=]|\.iso\b|\.tar\.gz\b|\.tgz\b|\.zip\b|\.deb\b|\.rpm\b|"
    r"\.sh\b.*\|\s*(ba)?sh|pip install|gem install|npm install)",
    re.I,
)
_INTEGRITY_OFF = re.compile(
    r"(gpgcheck\s*[:=]\s*(no|false|0|'0'|\"0\")|validate_certs\s*[:=]\s*(no|false)|sslverify\s*[:=]\s*(no|false|0)|"
    r"--no-check-certificate|--insecure|\bcurl\s+(-\w*k\b)|verify_ssl\s*[:=]\s*(no|false)|checksum\s*[:=]\s*(none|no|false)|"
    r"allow_unauthenticated|--allow-unauthenticated|trusted\s*=\s*yes|ignore_errors.*gpg)",
    re.I,
)
_CHECKSUM = re.compile(r"(checksum|sha256|sha512|gpgkey|gpg_key|signature|\bsha\b|digest)", re.I)

_WEAK_CRYPTO = re.compile(
    r"\b(md5|sha-?1|des|3des|des3|rc4|arcfour|sslv2|sslv3|tlsv1(?:\.0)?(?![.\d])|tls1_0|blowfish|crypt\(|ssha)\b",
    re.I,
)

_PUPPET_CASE = re.compile(r"(\bcase\b[^{]*\{|\?\s*\{)")
_RUBY_CASE = re.compile(r"^\s*(\w+\s*=\s*)?case\b")
_RUBY_OPEN = re.compile(r"^\s*(if|unless|while|until|case|begin|def|class|module)\b|\bdo(\s*\|[^|]*\|)?\s*$")
_RUBY_INLINE_IF = re.compile(r"\S.*\b(if|unless|while|until)\b")


@dataclass(frozen=True)
class Candidate:
    line: int
    smell: str
    rule: str


def _is_comment(tech: str, text: str) -> bool:
    return bool(_COMMENT[tech].search(text))


def _strip_comment(text: str) -> str:
    # remove trailing "# ..." that is not inside a quoted string (approximate)
    out, quote = [], None
    for ch in text:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out)


def _has_literal_value(code: str) -> bool:
    match = _ASSIGN.search(code)
    if not match:
        return bool(_QUOTED.search(code))
    value = match.group(3)
    if _VAR_REF.search(code[match.start():]):
        return False
    return value.lower() not in {"true", "false", "yes", "no", "nil", "undef", "none", "null", "present", "absent"}


def _line_candidates(tech: str, text: str) -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    if not text.strip():
        return found
    comment = _is_comment(tech, text) and text.strip().startswith(("#", "//", "/*", "<%#", "{#"))
    code = text if comment else _strip_comment(text)

    if comment:
        if _SUSP_WORDS.search(text):
            found.append(("Suspicious comment", "susp_words"))
        return found  # remaining smells concern active configuration
    if "#" in text and _SUSP_WORDS.search(text[text.find("#"):]):
        found.append(("Suspicious comment", "trailing_comment"))

    if _ADMIN_FLAG.search(code):
        found.append(("Admin by default", "admin_flag"))
    elif _ADMIN_TOKEN.search(code) and _USER_CTX.search(code):
        found.append(("Admin by default", "admin_token_user_ctx"))

    if _EMPTY_ASSIGN.search(code) or _EMPTY_FLAG.search(code):
        found.append(("Empty password", "empty_value"))

    if _URL_CRED.search(code) or (_CMD_PWD.search(code) and re.search(r"(mysql|psql|mongo|redis-cli|curl|wget|sshpass|htpasswd)", code, re.I)):
        found.append(("Hard-coded secret", "inline_credential"))
    elif _SECRET_NAME.search(code) and _has_literal_value(code):
        found.append(("Hard-coded secret", "secret_name_literal"))

    if _INVALID_BIND.search(code):
        found.append(("Unrestricted IP Address", "any_addr" if _BIND_CTX.search(code) else "any_addr_noctx"))

    if _HTTP.search(code):
        found.append(("Use of HTTP without SSL/TLS", "http_scheme"))

    if _INTEGRITY_OFF.search(code):
        found.append(("No integrity check", "integrity_disabled"))

    if _WEAK_CRYPTO.search(code):
        found.append(("Use of weak cryptography algorithms", "weak_primitive"))
    return found


def _download_without_checksum(tech: str, lines: Sequence[str]) -> List[Tuple[int, str]]:
    """Download-like lines whose surrounding block (+/-6 lines) never mentions a checksum/signature."""
    out = []
    for idx, text in enumerate(lines):
        if not text.strip() or text.strip().startswith("#"):
            continue
        if _DOWNLOAD.search(text) and re.search(r"(https?://|ftp://|\burl\b|source|get_url|remote_file|wget|curl)", text, re.I):
            lo, hi = max(0, idx - 6), min(len(lines), idx + 7)
            block = "\n".join(lines[lo:hi])
            if not _CHECKSUM.search(block):
                out.append((idx + 1, "download_no_checksum"))
    return out


def _missing_default(tech: str, lines: Sequence[str]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    n = len(lines)
    if tech == "puppet":
        for idx, text in enumerate(lines):
            code = _strip_comment(text)
            if not _PUPPET_CASE.search(code):
                continue
            depth, seen_default = 0, False
            started = False
            for j in range(idx, min(n, idx + 400)):
                seg = _strip_comment(lines[j])
                if j > idx and re.search(r"\bdefault\b", seg):
                    seen_default = True
                if j == idx:
                    seg = seg[_PUPPET_CASE.search(seg).start():]
                for ch in seg:
                    if ch == "{":
                        depth += 1
                        started = True
                    elif ch == "}":
                        depth -= 1
                if started and depth <= 0:
                    break
            if not seen_default:
                out.append((idx + 1, "puppet_case_no_default"))
    elif tech == "chef":
        for idx, text in enumerate(lines):
            if not _RUBY_CASE.search(_strip_comment(text)):
                continue
            depth, has_else = 1, False
            for j in range(idx + 1, min(n, idx + 400)):
                seg = _strip_comment(lines[j]).strip()
                if not seg:
                    continue
                if _RUBY_OPEN.search(seg) and not _RUBY_INLINE_IF.match(seg):
                    depth += 1
                if re.match(r"^end\b", seg):
                    depth -= 1
                    if depth == 0:
                        break
                if depth == 1 and re.match(r"^else\b", seg):
                    has_else = True
            if not has_else:
                out.append((idx + 1, "ruby_case_no_else"))
    return out


def generate_candidates(
    tech: str,
    lines: Sequence[str],
    exclude: Optional[Set[Tuple[int, str]]] = None,
    smells: Optional[Iterable[str]] = None,
) -> List[Candidate]:
    """All relaxed candidates in a script, minus ``exclude`` (stage-1 (line, smell) pairs)."""
    exclude = exclude or set()
    allowed = set(smells) if smells else None
    seen: Dict[Tuple[int, str], Candidate] = {}

    def add(line: int, smell: str, rule: str) -> None:
        key = (line, smell)
        if key in exclude or key in seen:
            return
        if allowed is not None and smell not in allowed:
            return
        seen[key] = Candidate(line=line, smell=smell, rule=rule)

    for idx, text in enumerate(lines, start=1):
        for smell, rule in _line_candidates(tech, text):
            add(idx, smell, rule)
    for line, rule in _download_without_checksum(tech, lines):
        add(line, "No integrity check", rule)
    for line, rule in _missing_default(tech, lines):
        add(line, "Missing Default in Case Statement", rule)
    return sorted(seen.values(), key=lambda c: (c.line, c.smell))
