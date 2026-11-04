"""Canonical security-smell vocabulary, aliases and teacher-facing definitions.

Category normalisation reproduces IntelliSA's ``replication/RQ1/code/evaluate.py`` exactly,
so detections produced here are scored with the same rules as the published baselines.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# Display order used by IntelliSA (Table 7) -------------------------------------------
SMELLS: List[str] = [
    "Admin by default",
    "Empty password",
    "Hard-coded secret",
    "Unrestricted IP Address",
    "Suspicious comment",
    "Use of HTTP without SSL/TLS",
    "No integrity check",
    "Use of weak cryptography algorithms",
    "Missing Default in Case Statement",
]

# The four smells IntelliSA routes through its neural filter (G3 ablation)
INTELLISA_FILTERED: List[str] = [
    "Hard-coded secret",
    "Suspicious comment",
    "Use of HTTP without SSL/TLS",
    "Use of weak cryptography algorithms",
]

SMELL_SHORT: Dict[str, str] = {
    "Admin by default": "admin",
    "Empty password": "empty_pwd",
    "Hard-coded secret": "hardcoded_secret",
    "Unrestricted IP Address": "invalid_bind",
    "Suspicious comment": "suspicious_comment",
    "Use of HTTP without SSL/TLS": "http",
    "No integrity check": "no_integrity",
    "Use of weak cryptography algorithms": "weak_crypto",
    "Missing Default in Case Statement": "missing_default",
}

SMELL_DISPLAY: Dict[str, str] = {
    "Admin by default": "Admin by default",
    "Empty password": "Empty password",
    "Hard-coded secret": "Hard-coded secret",
    "Unrestricted IP Address": "Invalid IP address binding",
    "Suspicious comment": "Suspicious comment",
    "Use of HTTP without SSL/TLS": "Use of HTTP without TLS",
    "No integrity check": "No integrity check",
    "Use of weak cryptography algorithms": "Use of weak crypto alg.",
    "Missing Default in Case Statement": "Missing default case statement",
}

_ALIASES: Dict[str, Optional[str]] = {
    "admin by default": "Admin by default",
    "admin-by-default": "Admin by default",
    "empty password": "Empty password",
    "empty-password": "Empty password",
    "hard coded secret": "Hard-coded secret",
    "hard-coded secret": "Hard-coded secret",
    "hardcoded secret": "Hard-coded secret",
    "hardcoded-password": "Hard-coded secret",
    "hardcoded password": "Hard-coded secret",
    "hardcoded-username": "Hard-coded secret",
    "missing default in case statement": "Missing Default in Case Statement",
    "missing default case statement": "Missing Default in Case Statement",
    "missing default switch": "Missing Default in Case Statement",
    "missing default in switch statement": "Missing Default in Case Statement",
    "missing default in case statements": "Missing Default in Case Statement",
    "no integrity check": "No integrity check",
    "no-integrity-check": "No integrity check",
    "suspicious comment": "Suspicious comment",
    "unrestricted ip address": "Unrestricted IP Address",
    "improper ip address binding": "Unrestricted IP Address",
    "use of http": "Use of HTTP without SSL/TLS",
    "use of http without ssl/tls": "Use of HTTP without SSL/TLS",
    "http without ssl/tls": "Use of HTTP without SSL/TLS",
    "use of weak cryptography algorithms": "Use of weak cryptography algorithms",
    "weak cryptography algorithms": "Use of weak cryptography algorithms",
    "weak cryptography algorithm": "Use of weak cryptography algorithms",
    "weak cryptography": "Use of weak cryptography algorithms",
    "none": None,
    "no issues found": None,
    "no issue": None,
    "no-issue": None,
    "no issues": None,
    "n/a": None,
}

# Extra spellings that occur in GLITCH outputs, GLITCH error codes and IntelliSA training data.
# These only extend the alias table; they never change how an IntelliSA category is mapped.
_EXTRA_ALIASES: Dict[str, Optional[str]] = {
    "hardcoded user": "Hard-coded secret",
    "hard-coded password": "Hard-coded secret",
    "hard-coded user": "Hard-coded secret",
    "hard coded secret": "Hard-coded secret",
    "invalid ip address binding": "Unrestricted IP Address",
    "invalid-bind": "Unrestricted IP Address",
    "use of http without tls": "Use of HTTP without SSL/TLS",
    "weak crypto algorithm": "Use of weak cryptography algorithms",
    "weak-crypto": "Use of weak cryptography algorithms",
    "use of weak crypto alg.": "Use of weak cryptography algorithms",
    "sec def admin": "Admin by default",
    "sec empty pass": "Empty password",
    "sec hard secr": "Hard-coded secret",
    "sec hard pass": "Hard-coded secret",
    "sec hard user": "Hard-coded secret",
    "sec invalid bind": "Unrestricted IP Address",
    "sec susp comm": "Suspicious comment",
    "sec https": "Use of HTTP without SSL/TLS",
    "sec no int check": "No integrity check",
    "sec weak crypt": "Use of weak cryptography algorithms",
    "sec no default switch": "Missing Default in Case Statement",
    "hard coded secret": "Hard-coded secret",
    "use of weak cryptography algorithms": "Use of weak cryptography algorithms",
    "use of http without ssl tls": "Use of HTTP without SSL/TLS",
    "weak cryptography algorithms": "Use of weak cryptography algorithms",
}


def normalize_category(value: Optional[str]) -> Optional[str]:
    """Map a raw category string to a canonical smell (``None`` for 'no smell').

    Identical to IntelliSA's ``normalize_category`` for every string it knows, plus the
    GLITCH/training spellings listed in ``_EXTRA_ALIASES``. Unknown strings are returned unchanged
    (and therefore never match the oracle), as in IntelliSA.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    key = raw.lower()
    if key in _ALIASES:
        return _ALIASES[key]
    key = key.replace("-", " ").replace("_", " ")
    key = re.sub(r"\s+", " ", key).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    if key in _EXTRA_ALIASES:
        return _EXTRA_ALIASES[key]
    key2 = key.replace("/", " ")
    key2 = re.sub(r"\s+", " ", key2).strip()
    if key2 in _EXTRA_ALIASES:
        return _EXTRA_ALIASES[key2]
    if raw in SMELLS:
        return raw
    return raw


def is_canonical(value: Optional[str]) -> bool:
    return value in SMELLS


# ------------------------------------------------------------------ teacher definitions
# Formal GLITCH rules (Saavedra & Ferreira, ASE 2022) as used in IntelliSA's prompts.
RULES: Dict[str, str] = {
    "Admin by default": "(isAttribute(x) ∨ isVariable(x)) ∧ (isUser(x.name) ∨ isRole(x.name)) ∧ isAdmin(x.value) ∧ ¬x.has_variable",
    "Empty password": "(isAttribute(x) ∨ isVariable(x)) ∧ isPassword(x.name) ∧ length(x.value) == 0",
    "Hard-coded secret": "(isAttribute(x) ∨ isVariable(x)) ∧ (isPassword(x.name) ∨ isSecret(x.name) ∨ isUser(x.name)) ∧ ¬x.has_variable",
    "Unrestricted IP Address": "(isAttribute(x) ∨ isVariable(x)) ∧ isInvalidBind(x.value)",
    "Suspicious comment": "isComment(x) ∧ hasWrongWords(x.content)",
    "Use of HTTP without SSL/TLS": "(isAttribute(x) ∨ isVariable(x)) ∧ isURL(x.value) ∧ hasHTTP(x.value) ∧ ¬hasHTTPWhitelist(x.value)",
    "No integrity check": "(isAtomicUnit(x) ∧ hasDownload(x.attributes) ∧ ¬hasChecksum(x.attributes)) ∨ ((isAttribute(x) ∨ isVariable(x)) ∧ isCheckSum(x.name) ∧ (x.value == \"no\" ∨ x.value == \"false\"))",
    "Use of weak cryptography algorithms": "(isAttribute(x) ∨ isVariable(x)) ∧ isWeakCrypt(x.value) ∧ ¬hasWeakCryptWhitelist(x.name) ∧ ¬hasWeakCryptWhitelist(x.value)",
    "Missing Default in Case Statement": "isConditionStatement(x) ∧ x.is_default == False ∧ ¬isDefault(x.else_statement)",
}

KEYWORDS: Dict[str, str] = {
    "Admin by default": 'isUser(): "user", "uname", "username", "login", "userid", "loginid"; isAdmin(): "admin", "root"',
    "Empty password": 'isPassword(): "pass", "pwd", "password", "passwd", "passno", "pass-no"',
    "Hard-coded secret": 'isUser(): "user", "uname", "username", "login", "userid", "loginid"; isPassword(): "pass", "pwd", "password", "passwd", "passno", "pass-no"; isSecret(): "auth_token", "authentication_token", "secret", "ssh_key"',
    "Unrestricted IP Address": 'isInvalidBind(): "0.0.0.0"',
    "Suspicious comment": 'hasWrongWords(): "bug", "debug", "todo", "hack", "solve", "fixme"',
    "Use of HTTP without SSL/TLS": 'hasHTTP(): "http"; hasHTTPWhitelist(): "localhost", "127.0.0.1"',
    "No integrity check": 'isDownload(): URLs ending in ".iso", ".tar.gz", ".tgz", ".zip", ".deb", ".rpm"; isCheckSum(): "gpg", "checksum"',
    "Use of weak cryptography algorithms": 'isWeakCrypt(): "md5", "sha1", "arcfour"; hasWeakCryptWhitelist(): "checksum"',
    "Missing Default in Case Statement": "isConditionStatement(): case/switch/selector constructs; isDefault(): default/else branches",
}

# Natural-language definitions. The four filtered smells reuse IntelliSA's released
# ``smell_definitions.yaml``; the other five follow Rahman et al. (ICSE'19, TOSEM'21).
DEFINITIONS: Dict[str, str] = {
    "Admin by default": (
        "Admin by default: a user, role or login is configured with administrative privileges "
        "(e.g. 'root', 'admin', 'sa') as a literal default, violating least privilege. This includes "
        "commands or connection settings that log in as root/admin (e.g. 'mysql -u root'). "
        "NOT admin by default: values coming from variables/attributes the operator sets, "
        "checks against root, or non-security uses of the word."
    ),
    "Empty password": (
        "Empty password: a password, passphrase or credential attribute is assigned a zero-length "
        "string (e.g. password: '' or -p'' in a command). NOT empty password: unset/undefined values "
        "filled at runtime, or variables whose value comes from elsewhere."
    ),
    "Hard-coded secret": (
        "A hard-coded secret is sensitive information such as passwords, API keys, tokens, certificates, "
        "usernames of privileged accounts or other credentials directly embedded in the code as literal "
        "values, rather than retrieved from external configuration, environment variables or a secret "
        "manager. NOT hard-coded secrets: placeholders (e.g. 'your-api-key-here', 'REPLACE_ME'), example "
        "credentials in comments, variable names that suggest secrets but hold non-sensitive values, and "
        "parameter names without actual secret values."
    ),
    "Unrestricted IP Address": (
        "Invalid IP address binding: a service is bound to 0.0.0.0 (or ::), exposing it on every network "
        "interface. NOT this smell: binding to specific addresses, documentation, or 0.0.0.0/0 used as a "
        "deny rule."
    ),
    "Suspicious comment": (
        "A suspicious comment indicates potential security issues, incomplete security implementations or "
        "areas requiring security attention (TODO/FIXME/HACK/BUG about security, disabled checks, temporary "
        "insecure workarounds, hard-coded values to be externalised). NOT suspicious: comments about "
        "functional bugs, performance or features unrelated to security, and plain documentation."
    ),
    "Use of HTTP without SSL/TLS": (
        "Use of HTTP without SSL/TLS: a network resource is configured or invoked over unencrypted http:// "
        "(urls, repositories, mirrors, download sources, endpoints, callbacks) or TLS is explicitly disabled. "
        "NOT this smell: https URLs, bare domains, localhost/127.0.0.1, integrity-verified resources, and "
        "comments or strings that mention http without using it as an active configuration value."
    ),
    "No integrity check": (
        "No integrity check: content (packages, archives, ISOs, scripts, keys) is downloaded or installed "
        "from a remote source without verifying a checksum or signature, or such verification is explicitly "
        "disabled (e.g. gpgcheck: no, validate_certs/checksum false). NOT this smell: downloads that set a "
        "checksum/GPG key, or local files."
    ),
    "Use of weak cryptography algorithms": (
        "Use of weak cryptography algorithms: broken or deprecated primitives (MD5, SHA1, DES, 3DES, RC4, "
        "arcfour, SSLv2/3, TLS1.0) used for security purposes such as password hashing, signing or "
        "encryption. NOT this smell: weak hashes used as non-security checksums, mentions in comments/docs, "
        "or variable names without an actual use."
    ),
    "Missing Default in Case Statement": (
        "Missing default case statement: a case/switch/selector (e.g. Puppet case or ? {}, Ruby case/when, "
        "Ansible conditional chains) does not handle the default branch, so unexpected values are not "
        "handled. NOT this smell: constructs that include a default/else branch."
    ),
}

# IntelliSA training-data smell ids -> canonical
INTELLISA_TRAIN_SMELLS: Dict[str, str] = {
    "hard_coded_secret": "Hard-coded secret",
    "suspicious_comment": "Suspicious comment",
    "use_of_http": "Use of HTTP without SSL/TLS",
    "use_of_http_without_ssl_tls": "Use of HTTP without SSL/TLS",
    "weak_cryptography_algorithms": "Use of weak cryptography algorithms",
    "use_of_weak_cryptography_algorithms": "Use of weak cryptography algorithms",
}


def smells_to_verify(mode: str) -> List[str]:
    if mode == "all":
        return list(SMELLS)
    if mode == "intellisa4":
        return list(INTELLISA_FILTERED)
    raise ValueError(f"Unknown verify_smells mode {mode!r} (expected 'all' or 'intellisa4')")
