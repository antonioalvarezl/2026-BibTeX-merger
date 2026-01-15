#!/usr/bin/env python3
"""Merge duplicate BibTeX entries and re-key them to NameYearKeyword."""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Entry:
    entry_type: str
    key: str
    fields: Dict[str, str]


@dataclass
class MergedItem:
    entry: Entry
    old_ids: List[str]


PRESERVE_TYPES = {"comment", "preamble", "string"}
PUBLISHED_TYPES = {
    "article",
    "inproceedings",
    "proceedings",
    "book",
    "incollection",
    "inbook",
    "phdthesis",
    "mastersthesis",
    "techreport",
}
PUBLICATION_FIELDS = {
    "journal",
    "booktitle",
    "publisher",
    "volume",
    "number",
    "pages",
    "edition",
    "address",
    "school",
    "institution",
    "organization",
}
ARXIV_OK_FIELDS = {
    "year",
    "month",
    "journal",
    "booktitle",
    "publisher",
    "volume",
    "number",
    "pages",
    "note",
    "url",
    "eprint",
    "archiveprefix",
    "primaryclass",
    "howpublished",
    "organization",
    "institution",
}
FIELD_ORDER = [
    "author",
    "title",
    "journaltitle",
    "booktitle",
    "year",
    "month",
    "volume",
    "number",
    "pages",
    "publisher",
    "edition",
    "series",
    "school",
    "institution",
    "organization",
    "doi",
    "eprint",
    "eprinttype",
    "eprintclass",
    "url",
    "urldate",
    "note",
    "keywords",
    "isbn",
    "ids",
]
STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "without",
    "from",
    "by",
    "at",
    "into",
    "via",
    "using",
    "use",
    "is",
    "are",
    "be",
    "as",
    "we",
    "our",
    "their",
}
AUTHOR_MARKERS = {
    "others",
    "et al",
    "et al.",
}
NAME_PARTICLES = {
    "al",
    "bin",
    "da",
    "das",
    "de",
    "del",
    "der",
    "di",
    "dos",
    "du",
    "el",
    "la",
    "le",
    "van",
    "von",
    "y",
}
INSTITUTION_KEYWORDS = {
    "academy",
    "association",
    "center",
    "centre",
    "college",
    "committee",
    "consortium",
    "corporation",
    "department",
    "foundation",
    "group",
    "institute",
    "laboratory",
    "laboratories",
    "lab",
    "press",
    "publishers",
    "school",
    "society",
    "team",
    "university",
}

LATEX_ACCENT_RE = re.compile(r"\\[`'^\"~=.uvHtcdbkr]{\s*([A-Za-z])\s*}")
LATEX_ACCENT_SIMPLE_RE = re.compile(r"\\[`'^\"~=.uvHtcdbkr]([A-Za-z])")
LATEX_ACCENT_BRACED_RE = re.compile(
    r"\{\\([`'^\"~=.uvHtcdbkr])\s*(\\[ij]|[A-Za-z])\s*\}"
)
LATEX_ACCENT_CMD_RE = re.compile(
    r"\\([`'^\"~=.uvHtcdbkr])\s*\{\s*(\\[ij]|[A-Za-z])\s*\}"
)
LATEX_ACCENT_CMD_SIMPLE_RE = re.compile(
    r"\\([`'^\"~=.uvHtcdbkr])(\\[ij]|[A-Za-z])"
)
LATEX_LIGATURES = {
    r"\ss": "ss",
    r"\ae": "ae",
    r"\AE": "AE",
    r"\oe": "oe",
    r"\OE": "OE",
    r"\aa": "aa",
    r"\AA": "AA",
    r"\o": "o",
    r"\O": "O",
    r"\l": "l",
    r"\L": "L",
    r"\i": "i",
    r"\j": "j",
}
LATEX_COMBINING_MARKS = {
    "\u0300": r"\`",
    "\u0301": r"\'",
    "\u0302": r"\^",
    "\u0303": r"\~",
    "\u0304": r"\=",
    "\u0306": r"\u",
    "\u0307": r"\.",
    "\u0308": r"\"",
    "\u030a": r"\r",
    "\u030b": r"\H",
    "\u030c": r"\v",
    "\u0327": r"\c",
    "\u0328": r"\k",
    "\u0331": r"\b",
}
LATEX_ACCENT_TO_COMBINING = {
    "`": "\u0300",
    "'": "\u0301",
    "^": "\u0302",
    "~": "\u0303",
    "=": "\u0304",
    "u": "\u0306",
    ".": "\u0307",
    "\"": "\u0308",
    "r": "\u030a",
    "H": "\u030b",
    "v": "\u030c",
    "c": "\u0327",
    "k": "\u0328",
    "b": "\u0331",
    "d": "\u0323",
}
LATEX_SPECIAL_CHARS = {
    "\u00df": r"\ss",
    "\u00e6": r"\ae",
    "\u00c6": r"\AE",
    "\u0153": r"\oe",
    "\u0152": r"\OE",
    "\u00f8": r"\o",
    "\u00d8": r"\O",
    "\u0142": r"\l",
    "\u0141": r"\L",
    "\u0111": r"\dj",
    "\u0110": r"\DJ",
    "\u0131": r"\i",
    "\u0237": r"\j",
}
LATEX_COMMAND_TO_UNICODE = {value: key for key, value in LATEX_SPECIAL_CHARS.items()}
LATEX_COMMAND_BRACED_RE = re.compile(r"\{\\([A-Za-z]+)\}")
LATEX_SKIP_FIELDS = {
    "url",
    "doi",
    "eprint",
    "archiveprefix",
    "primaryclass",
    "eprinttype",
    "eprintclass",
    "urldate",
    "ids",
}
MONTHS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}
CONTAINMENT_FIELDS = {
    "publisher",
    "organization",
    "institution",
    "address",
    "location",
    "series",
    "note",
}
SOFT_CONFLICT_FIELDS = {
    "month",
    "volume",
    "number",
    "pages",
    "publisher",
}
FIELD_MARKER_RE = re.compile(
    r"\b(author|title|journal|journaltitle|booktitle|year|month|volume|number|pages|publisher|"
    r"doi|url|eprint|eprinttype|eprintclass|archiveprefix|primaryclass|edition|urldate)\s*="
)


def has_balanced_outer_braces(value: str) -> bool:
    if not (value.startswith("{") and value.endswith("}")):
        return False
    depth = 0
    for idx, ch in enumerate(value):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and idx != len(value) - 1:
                return False
    return depth == 0


def strip_outer_braces_quotes(value: str) -> str:
    if value is None:
        return ""
    text = value.strip()
    if not text:
        return ""
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    if has_balanced_outer_braces(text):
        text = text[1:-1].strip()
    return text


def strip_latex(text: str) -> str:
    for latex, repl in LATEX_LIGATURES.items():
        text = text.replace(latex, repl)
    text = text.replace(r"\&", "and")
    text = LATEX_ACCENT_BRACED_RE.sub(r"\2", text)
    text = LATEX_ACCENT_CMD_RE.sub(r"\2", text)
    text = LATEX_ACCENT_RE.sub(r"\1", text)
    text = LATEX_ACCENT_SIMPLE_RE.sub(r"\1", text)
    text = LATEX_ACCENT_CMD_SIMPLE_RE.sub(r"\2", text)
    return text


def latex_to_unicode(text: str) -> str:
    if not text or "\\" not in text:
        return text

    def replace_command_braced(match: re.Match) -> str:
        cmd = f"\\{match.group(1)}"
        return LATEX_COMMAND_TO_UNICODE.get(cmd, match.group(0))

    def replace_accent(match: re.Match, source: str) -> str:
        cmd = match.group(1)
        token = match.group(2)
        letter = token[1:] if token in {r"\i", r"\j"} else token
        mark = LATEX_ACCENT_TO_COMBINING.get(cmd)
        if not mark:
            return match.group(0)
        idx = match.start()
        prev = source[idx - 1] if idx > 0 else ""
        if prev and prev.isalnum():
            letter = letter.lower()
        return unicodedata.normalize("NFC", letter + mark)

    text = LATEX_ACCENT_BRACED_RE.sub(lambda m: replace_accent(m, text), text)
    text = LATEX_ACCENT_CMD_RE.sub(lambda m: replace_accent(m, text), text)
    text = LATEX_ACCENT_CMD_SIMPLE_RE.sub(lambda m: replace_accent(m, text), text)
    text = LATEX_COMMAND_BRACED_RE.sub(replace_command_braced, text)
    for cmd, uni in LATEX_COMMAND_TO_UNICODE.items():
        text = text.replace(cmd, uni)
    return unicodedata.normalize("NFC", text)


def strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def is_institution_name(name: str) -> bool:
    stripped = strip_outer_braces_quotes(name).strip()
    if not stripped:
        return False
    if "," in stripped:
        return False
    words = [word.lower() for word in re.findall(r"[A-Za-z]+", stripped)]
    if not words:
        return False
    return any(word in INSTITUTION_KEYWORDS for word in words)


def normalize_name_token(token: str) -> str:
    if not token:
        return token
    if "\\" in token or "{" in token or "}" in token:
        return token
    lower = token.lower()
    if lower in NAME_PARTICLES:
        return lower
    if token.isupper() or token.islower():
        return token[:1].upper() + token[1:].lower()
    return token


def normalize_name_text(text: str) -> str:
    text = latex_to_unicode(text)
    text = text.replace("{", "").replace("}", "").replace("\\", "")
    parts = re.split(r"(\s+)", text.strip())
    normalized_parts: List[str] = []
    for part in parts:
        if not part or part.isspace():
            normalized_parts.append(part)
            continue
        subparts = re.split(r"([-'’])", part)
        normalized_subparts: List[str] = []
        for sub in subparts:
            if sub in {"-", "'", "’"}:
                normalized_subparts.append(sub)
                continue
            normalized_subparts.append(normalize_name_token(sub))
        normalized_parts.append("".join(normalized_subparts))
    return "".join(normalized_parts).strip()


def format_person_name(name: str) -> str:
    name = normalize_whitespace(name)
    if not name:
        return ""
    if has_balanced_outer_braces(name):
        inner = strip_outer_braces_quotes(name)
        if is_institution_name(inner):
            return f"{{{{{inner}}}}}"
        name = inner
    if is_institution_name(name):
        return f"{{{{{name}}}}}"
    if "," in name:
        last, rest = name.split(",", 1)
        last = normalize_name_text(last)
        rest = normalize_name_text(rest)
        rest = rest.strip()
        return f"{last}, {rest}".strip(", ").strip()
    tokens = name.split()
    if len(tokens) == 1:
        return normalize_name_text(name)
    last_tokens = [tokens[-1]]
    idx = len(tokens) - 2
    while idx >= 0 and tokens[idx].lower() in NAME_PARTICLES:
        last_tokens.insert(0, tokens[idx])
        idx -= 1
    first_tokens = tokens[: idx + 1]
    last = normalize_name_text(" ".join(last_tokens))
    first = normalize_name_text(" ".join(first_tokens))
    return f"{last}, {first}".strip()


def is_author_marker(value: str) -> bool:
    return normalize_string(value) in AUTHOR_MARKERS


def format_author_list(value: str) -> str:
    authors = split_authors(value)
    if not authors:
        return ""
    formatted: List[str] = []
    for author in authors:
        if is_author_marker(author):
            continue
        formatted_name = format_person_name(author)
        if formatted_name:
            formatted.append(formatted_name)
    return " and ".join(formatted)


def should_protect_title_token(token: str) -> bool:
    if not token or len(token) < 2:
        return False
    if any(ch in token for ch in "{}\\"):
        return False
    if token.isupper():
        return True
    if re.search(r"[A-Z].*[A-Z]", token):
        return True
    if re.search(r"[A-Z]", token) and re.search(r"\d", token):
        return True
    return False


def protect_title_acronyms(title: str) -> str:
    if not title:
        return title

    def process_token(token: str) -> str:
        parts = re.split(r"([-/])", token)
        processed: List[str] = []
        for part in parts:
            if part in {"-", "/"}:
                processed.append(part)
                continue
            if should_protect_title_token(part):
                processed.append(f"{{{part}}}")
            else:
                processed.append(part)
        return "".join(processed)

    result: List[str] = []
    token: List[str] = []
    depth = 0
    in_command = False
    for ch in title:
        if in_command:
            result.append(ch)
            if not ch.isalpha():
                in_command = False
            continue
        if ch == "\\" and depth == 0:
            if token:
                result.append(process_token("".join(token)))
                token = []
            result.append(ch)
            in_command = True
            continue
        if ch == "{":
            if depth == 0 and token:
                result.append(process_token("".join(token)))
                token = []
            depth += 1
            result.append(ch)
            continue
        if ch == "}":
            if depth == 0 and token:
                result.append(process_token("".join(token)))
                token = []
            if depth > 0:
                depth -= 1
            result.append(ch)
            continue
        if depth > 0:
            result.append(ch)
            continue
        if ch.isalnum() or ch in "-/":
            token.append(ch)
        else:
            if token:
                result.append(process_token("".join(token)))
                token = []
            result.append(ch)
    if token:
        result.append(process_token("".join(token)))
    return "".join(result)


def format_pages(value: str) -> str:
    text = strip_outer_braces_quotes(value).strip()
    if not text:
        return ""
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    return re.sub(r"(?<=\w)-(?=\w)", "--", text)


def normalize_eprint_type(value: str) -> str:
    text = normalize_string(value)
    if not text:
        return ""
    if "arxiv" in text:
        return "arxiv"
    return text


def is_corrupted_field_value(value: str) -> bool:
    if not value:
        return False
    return bool(FIELD_MARKER_RE.search(value))


def should_latexify_field(field: str) -> bool:
    if field in LATEX_SKIP_FIELDS:
        return False
    if "url" in field or "file" in field:
        return False
    return True


def unicode_to_latex(text: str) -> str:
    result: List[str] = []
    for ch in text:
        if ord(ch) < 128:
            result.append(ch)
            continue
        if ch in LATEX_SPECIAL_CHARS:
            result.append(LATEX_SPECIAL_CHARS[ch])
            continue
        decomposed = unicodedata.normalize("NFD", ch)
        base = decomposed[0]
        combining = decomposed[1:]
        latex_cmd = None
        for mark in combining:
            latex_cmd = LATEX_COMBINING_MARKS.get(mark)
            if latex_cmd:
                break
        if latex_cmd and ord(base) < 128:
            result.append(f"{latex_cmd}{{{base}}}")
        else:
            fallback = strip_diacritics(ch)
            if fallback and ord(fallback[0]) < 128:
                result.append(fallback)
            else:
                result.append(ch)
    return "".join(result)


def escape_ampersands(text: str) -> str:
    return re.sub(r"(?<!\\)&", r"\\&", text)


def latexify_value(field: str, value: str) -> str:
    if not value or not should_latexify_field(field):
        return value
    text = unicode_to_latex(value)
    text = escape_ampersands(text)
    return text


def split_on_first_comma(text: str) -> Tuple[str, str]:
    depth = 0
    in_quotes = False
    escape = False
    for idx, ch in enumerate(text):
        if in_quotes:
            if ch == '"' and not escape:
                in_quotes = False
            escape = (ch == "\\") and not escape
            continue
        if ch == '"':
            in_quotes = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0 and not in_quotes:
            return text[:idx], text[idx + 1 :]
    return text, ""


def split_fields(text: str) -> List[str]:
    fields: List[str] = []
    current: List[str] = []
    depth = 0
    in_quotes = False
    escape = False
    for ch in text:
        if in_quotes:
            current.append(ch)
            if ch == '"' and not escape:
                in_quotes = False
            escape = (ch == "\\") and not escape
            continue
        if ch == '"':
            in_quotes = True
            current.append(ch)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(depth - 1, 0)
        if ch == "," and depth == 0 and not in_quotes:
            field = "".join(current).strip()
            if field:
                fields.append(field)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        fields.append(tail)
    return fields


def parse_entry_body(entry_type: str, body: str) -> Optional[Entry]:
    key_part, fields_part = split_on_first_comma(body)
    key = key_part.strip()
    if not key:
        return None
    fields: Dict[str, str] = {}
    for field in split_fields(fields_part):
        if "=" not in field:
            continue
        name, value = field.split("=", 1)
        name = name.strip().lower()
        if name == "journaltitle":
            name = "journal"
        elif name == "eprinttype":
            name = "archiveprefix"
        elif name == "eprintclass":
            name = "primaryclass"
        value = strip_outer_braces_quotes(value.strip())
        if name:
            fields[name] = value
    return Entry(entry_type=entry_type, key=key, fields=fields)


def parse_bibtex(text: str) -> Tuple[List[Entry], List[str]]:
    entries: List[Entry] = []
    passthrough: List[str] = []
    idx = 0
    while idx < len(text):
        at = text.find("@", idx)
        if at == -1:
            break
        parsed = parse_entry_at(text, at)
        if not parsed:
            break
        entry_type, body, end = parsed
        raw_block = text[at:end].strip()
        if entry_type in PRESERVE_TYPES:
            if raw_block:
                passthrough.append(raw_block)
        else:
            entry = parse_entry_body(entry_type, body)
            if entry:
                entries.append(entry)
        idx = end
    return entries, passthrough


def parse_entry_at(text: str, start: int) -> Optional[Tuple[str, str, int]]:
    idx = start + 1
    while idx < len(text) and text[idx].isspace():
        idx += 1
    type_start = idx
    while idx < len(text) and (text[idx].isalnum() or text[idx] in "_-"):
        idx += 1
    entry_type = text[type_start:idx].lower()
    while idx < len(text) and text[idx].isspace():
        idx += 1
    if idx >= len(text) or text[idx] not in "{(":
        return None
    open_char = text[idx]
    close_char = "}" if open_char == "{" else ")"
    idx += 1
    body_start = idx
    depth = 1
    while idx < len(text) and depth > 0:
        ch = text[idx]
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
        idx += 1
    if depth != 0:
        return None
    body_end = idx - 1
    body = text[body_start:body_end]
    return entry_type, body, idx


def normalize_string(value: str) -> str:
    text = strip_outer_braces_quotes(value)
    text = strip_latex(text)
    text = strip_diacritics(text)
    text = text.replace("&", " and ")
    text = text.replace("\n", " ")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"[\"'`]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def normalize_title(value: str) -> str:
    text = normalize_string(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_author(value: str) -> str:
    if is_corrupted_field_value(value):
        return ""
    authors = split_authors(value)
    authors, _ = strip_author_markers(authors)
    signatures = [author_signature(author) for author in authors]
    signatures = [sig for sig in signatures if sig]
    return " and ".join(signatures)


def normalize_doi(value: str) -> str:
    text = normalize_string(value)
    text = text.replace("doi:", "")
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "http://dx.doi.org/",
        "https://dx.doi.org/",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip()


def normalize_url(value: str) -> str:
    text = normalize_string(value)
    for prefix in ("http://", "https://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.rstrip("/")


def normalize_arxiv(value: str) -> str:
    text = normalize_string(value)
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace("arxiv:", "").replace("abs/", "").strip()
    text = re.sub(r"v\d+$", "", text)
    return text


def normalize_pages(value: str) -> str:
    text = strip_outer_braces_quotes(value)
    text = strip_latex(text)
    text = strip_diacritics(text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)
    text = text.replace("--", "-")
    return text.lower()


def normalize_keywords(value: str) -> str:
    parts = re.split(r"[;,]", value)
    tokens = [normalize_string(part) for part in parts]
    tokens = [token for token in tokens if token]
    tokens = sorted(set(tokens))
    return ",".join(tokens)


def normalize_name_like(value: str) -> str:
    text = normalize_string(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_month(value: str) -> str:
    text = normalize_string(value)
    if not text:
        return ""
    if text.isdigit():
        month_num = int(text)
        if 1 <= month_num <= 12:
            return f"{month_num:02d}"
    if text in MONTHS:
        return MONTHS[text]
    if len(text) >= 3 and text[:3] in MONTHS:
        return MONTHS[text[:3]]
    return text


def normalize_edition(value: str) -> str:
    match = re.search(r"\d+", value)
    if match:
        return match.group(0)
    return normalize_string(value)


def normalize_isbn_values(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[\s,;]+", value)
    results = []
    for part in parts:
        cleaned = re.sub(r"[^0-9Xx]+", "", part)
        if len(cleaned) in (10, 13):
            results.append(cleaned.upper())
    return list(dict.fromkeys(results))


def normalize_numeric(value: str) -> str:
    match = re.search(r"\d+", value)
    if match:
        return match.group(0).lstrip("0") or "0"
    return normalize_string(value)


def normalize_venue_tokens(value: str) -> List[str]:
    text = normalize_string(value)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in STOPWORDS]
    tokens = [tok for tok in tokens if len(tok) >= 3]
    return tokens


def tokens_subset(left: str, right: str) -> bool:
    tokens_left = normalize_name_like(left).split()
    tokens_right = normalize_name_like(right).split()
    if not tokens_left or not tokens_right:
        return False
    set_left = set(tokens_left)
    set_right = set(tokens_right)
    return set_left <= set_right or set_right <= set_left


def keywords_equivalent(left: str, right: str) -> bool:
    left_set = set(filter(None, normalize_keywords(left).split(",")))
    right_set = set(filter(None, normalize_keywords(right).split(",")))
    if not left_set or not right_set:
        return False
    return left_set <= right_set or right_set <= left_set


def isbn_equivalent(left: str, right: str) -> bool:
    left_set = set(normalize_isbn_values(left))
    right_set = set(normalize_isbn_values(right))
    if not left_set or not right_set:
        return False
    return left_set <= right_set or right_set <= left_set


def extract_year(fields: Dict[str, str]) -> str:
    year = fields.get("year", "").strip()
    if year:
        match = re.search(r"\d{4}", year)
        return match.group(0) if match else year
    date = fields.get("date", "").strip()
    if date:
        match = re.search(r"\d{4}", date)
        if match:
            return match.group(0)
    return ""


def has_publication_fields(fields: Dict[str, str]) -> bool:
    return any(fields.get(field) for field in PUBLICATION_FIELDS)


def is_arxiv(entry: Entry) -> bool:
    fields = entry.fields
    archive = normalize_string(fields.get("archiveprefix", ""))
    eprint_type = normalize_string(fields.get("eprinttype", ""))
    eprint = fields.get("eprint", "")
    if "arxiv" in archive or "arxiv" in eprint_type:
        return True
    if eprint and not has_publication_fields(fields):
        return True
    if entry.entry_type in {"misc", "unpublished", "preprint"} and eprint:
        return True
    return False


def is_published(entry: Entry) -> bool:
    if has_publication_fields(entry.fields):
        return True
    return entry.entry_type in PUBLISHED_TYPES and not is_arxiv(entry)


def normalize_field_value(field: str, value: str) -> str:
    if field == "doi":
        return normalize_doi(value)
    if field == "url":
        return normalize_url(value)
    if field in {"eprint", "arxivid"}:
        return normalize_arxiv(value)
    if field == "title":
        return normalize_title(value)
    if field in {"author", "editor"}:
        return normalize_author(value)
    if field == "year":
        match = re.search(r"\d{4}", value)
        return match.group(0) if match else normalize_string(value)
    if field == "month":
        return normalize_month(value)
    if field == "publisher":
        return normalize_name_like(value)
    if field == "edition":
        return normalize_edition(value)
    if field == "isbn":
        return ",".join(normalize_isbn_values(value))
    if field in {"volume", "number"}:
        return normalize_numeric(value)
    if field == "pages":
        return normalize_pages(value)
    if field == "keywords":
        return normalize_keywords(value)
    return normalize_string(value)


def split_authors(value: str) -> List[str]:
    if not value:
        return []
    return [
        part.strip()
        for part in re.split(r"\s+and\s+", value, flags=re.IGNORECASE)
        if part.strip()
    ]


def strip_author_markers(authors: List[str]) -> Tuple[List[str], bool]:
    if not authors:
        return authors, False
    marker = normalize_string(authors[-1])
    if marker in {"others", "et al", "et al."}:
        return authors[:-1], True
    return authors, False


def author_signature(author: str) -> str:
    last_tokens, initials = parse_author(author)
    if not last_tokens:
        return ""
    last = "".join(last_tokens)
    return f"{last}{initials}".lower()


def parse_author(author: str) -> Tuple[List[str], str]:
    text = strip_outer_braces_quotes(author)
    text = strip_latex(text)
    text = strip_diacritics(text)
    text = text.replace("{", "").replace("}", "")
    if "," in text:
        last_part, rest = text.split(",", 1)
        rest_tokens = re.findall(r"[A-Za-z0-9]+", rest)
    else:
        parts = re.findall(r"[A-Za-z0-9]+", text)
        if not parts:
            return [], ""
        last_part = parts[-1]
        rest_tokens = parts[:-1]
    last_tokens = [tok.lower() for tok in re.findall(r"[A-Za-z0-9]+", last_part) if tok]
    last_tokens = [tok for tok in last_tokens if tok not in NAME_PARTICLES]
    initials = "".join(token[0].lower() for token in rest_tokens if token and token.lower() not in NAME_PARTICLES)
    return last_tokens, initials


def last_name_matches(left: List[str], right: List[str]) -> bool:
    if not left or not right:
        return False
    left_full = "".join(left)
    right_full = "".join(right)
    if left_full == right_full:
        return True
    if left_full.endswith(right_full) or right_full.endswith(left_full):
        return True
    return left[-1] == right[-1]


def initials_match(left: str, right: str) -> bool:
    if not left or not right:
        return True
    if left == right:
        return True
    return left.startswith(right) or right.startswith(left)


def authors_equivalent(left: str, right: str) -> bool:
    if is_corrupted_field_value(left) or is_corrupted_field_value(right):
        return False
    left_authors, left_marker = strip_author_markers(split_authors(left))
    right_authors, right_marker = strip_author_markers(split_authors(right))

    left_list = [parse_author(author) for author in left_authors]
    right_list = [parse_author(author) for author in right_authors]
    left_list = [item for item in left_list if item[0]]
    right_list = [item for item in right_list if item[0]]

    def authors_match_prefix(short_list: List[Tuple[List[str], str]], long_list: List[Tuple[List[str], str]]) -> bool:
        if len(short_list) > len(long_list):
            return False
        for (left_last, left_init), (right_last, right_init) in zip(short_list, long_list):
            if not last_name_matches(left_last, right_last):
                return False
            if not initials_match(left_init, right_init):
                return False
        return True

    if len(left_list) != len(right_list):
        if left_marker:
            return authors_match_prefix(left_list, right_list)
        if right_marker:
            return authors_match_prefix(right_list, left_list)
        return False

    return authors_match_prefix(left_list, right_list)


def title_token_set(value: str) -> set[str]:
    normalized = normalize_title(value)
    return set(normalized.split()) if normalized else set()


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    intersection = left & right
    union = left | right
    return len(intersection) / len(union)


def author_last_name_set(value: str) -> set[str]:
    if not value or is_corrupted_field_value(value):
        return set()
    authors, _ = strip_author_markers(split_authors(value))
    last_names: List[str] = []
    for author in authors:
        last_tokens, _ = parse_author(author)
        if last_tokens:
            last_names.append("".join(last_tokens))
    return set(last_names)


def author_similarity(left: str, right: str) -> float:
    left_set = author_last_name_set(left)
    right_set = author_last_name_set(right)
    return jaccard_similarity(left_set, right_set)


def likely_same(entry_a: Entry, entry_b: Entry) -> bool:
    doi_a = normalize_doi(entry_a.fields.get("doi", ""))
    doi_b = normalize_doi(entry_b.fields.get("doi", ""))
    if doi_a and doi_b and doi_a == doi_b:
        return True

    title_a = entry_a.fields.get("title", "")
    title_b = entry_b.fields.get("title", "")
    tokens_a = title_token_set(title_a)
    tokens_b = title_token_set(title_b)
    if not tokens_a or not tokens_b:
        return False
    if jaccard_similarity(tokens_a, tokens_b) < 0.9:
        return False

    year_a = extract_year(entry_a.fields)
    year_b = extract_year(entry_b.fields)
    if year_a and year_b and year_a != year_b:
        return False

    author_a = entry_a.fields.get("author") or entry_a.fields.get("editor") or ""
    author_b = entry_b.fields.get("author") or entry_b.fields.get("editor") or ""
    if author_a and author_b:
        if is_corrupted_field_value(author_a) or is_corrupted_field_value(author_b):
            return True
        if authors_equivalent(author_a, author_b):
            return True
        if author_similarity(author_a, author_b) >= 0.8:
            return True
        return False

    return True


def venue_equivalent(left: str, right: str) -> bool:
    tokens_left = normalize_venue_tokens(left)
    tokens_right = normalize_venue_tokens(right)
    if not tokens_left or not tokens_right:
        return False
    if tokens_left == tokens_right:
        return True
    if len(tokens_left) <= len(tokens_right):
        small, large = tokens_left, tokens_right
    else:
        small, large = tokens_right, tokens_left
    matches = 0
    used = set()
    for token in small:
        for idx, other in enumerate(large):
            if idx in used:
                continue
            if other.startswith(token) or token.startswith(other):
                matches += 1
                used.add(idx)
                break
    ratio = matches / len(small)
    return ratio >= 0.6


def compare_entries(entry_a: Entry, entry_b: Entry) -> Tuple[bool, Dict[str, Tuple[str, str]], str]:
    conflicts: Dict[str, Tuple[str, str]] = {}
    fields = set(entry_a.fields) | set(entry_b.fields)
    fields.discard("ids")
    for field in fields:
        value_a = entry_a.fields.get(field, "")
        value_b = entry_b.fields.get(field, "")
        if not value_a or not value_b:
            continue
        if is_corrupted_field_value(value_a) or is_corrupted_field_value(value_b):
            continue
        if field in {"author", "editor"}:
            if authors_equivalent(value_a, value_b):
                continue
            conflicts[field] = (value_a, value_b)
            continue
        if field in {"journal", "booktitle"}:
            if venue_equivalent(value_a, value_b):
                continue
        if field == "month":
            if normalize_month(value_a) == normalize_month(value_b):
                continue
        if field == "edition":
            if normalize_edition(value_a) == normalize_edition(value_b):
                continue
        if field in {"volume", "number"}:
            if normalize_numeric(value_a) == normalize_numeric(value_b):
                continue
        if field == "isbn":
            if isbn_equivalent(value_a, value_b):
                continue
        if field == "keywords":
            if keywords_equivalent(value_a, value_b):
                continue
        if field in CONTAINMENT_FIELDS:
            if tokens_subset(value_a, value_b):
                continue
        norm_a = normalize_field_value(field, value_a)
        norm_b = normalize_field_value(field, value_b)
        if norm_a != norm_b:
            conflicts[field] = (value_a, value_b)
    if not conflicts:
        return True, conflicts, ""
    arxiv_merge = is_arxiv(entry_a) != is_arxiv(entry_b)
    if arxiv_merge:
        disallowed = [field for field in conflicts if field not in ARXIV_OK_FIELDS]
        if not disallowed:
            return True, conflicts, "arxiv_vs_published"
    if all(field in SOFT_CONFLICT_FIELDS for field in conflicts):
        return True, conflicts, "soft_fields"
    if likely_same(entry_a, entry_b):
        return True, conflicts, "likely_same"
    return False, conflicts, ""


def entry_signatures(entry: Entry) -> List[str]:
    fields = entry.fields
    sigs: List[str] = []
    doi = normalize_doi(fields.get("doi", ""))
    if doi:
        sigs.append(f"doi:{doi}")
    arxiv = normalize_arxiv(fields.get("eprint", "") or fields.get("arxivid", ""))
    if arxiv:
        sigs.append(f"arxiv:{arxiv}")
    title = normalize_title(fields.get("title", ""))
    if title:
        year = extract_year(fields)
        if year:
            sigs.append(f"titleyear:{title}|{year}")
        else:
            sigs.append(f"title:{title}")
        author_last = normalize_string(first_author_last_name(entry))
        if author_last:
            sigs.append(f"titleauthoryear:{title}|{author_last}|{year}")
    return list(dict.fromkeys(sigs))


def first_author_last_name(entry: Entry) -> str:
    fields = entry.fields
    name_value = (
        fields.get("author")
        or fields.get("editor")
        or fields.get("organization")
        or fields.get("institution")
        or fields.get("publisher")
        or ""
    )
    if not name_value:
        return "Anon"
    name_value = strip_outer_braces_quotes(name_value)
    authors = re.split(r"\s+and\s+", name_value, flags=re.IGNORECASE)
    first = authors[0].strip()
    if "," in first:
        last = first.split(",", 1)[0].strip()
    else:
        parts = first.split()
        last = parts[-1] if parts else first
    return last or "Anon"


def clean_key_part(text: str) -> str:
    text = strip_outer_braces_quotes(text)
    text = strip_latex(text)
    text = strip_diacritics(text)
    text = re.sub(r"{([A-Za-z])}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = text.strip().title().replace(" ", "")
    return text or "Anon"


def title_keyword(title: str) -> str:
    if not title:
        return "Work"
    normalized = normalize_title(title)
    words = [word for word in normalized.split() if word]
    for word in words:
        if word not in STOPWORDS:
            return word.title()
    return words[0].title() if words else "Work"


def entry_score(entry: Entry) -> int:
    score = 0
    if is_published(entry):
        score += 100
    if entry.fields.get("doi"):
        score += 20
    if not is_arxiv(entry):
        score += 10
    score += sum(
        1
        for field, value in entry.fields.items()
        if value and field != "ids" and not is_corrupted_field_value(value)
    )
    if entry.entry_type in PUBLISHED_TYPES:
        score += 10
    return score


def build_base_key(fields: Dict[str, str]) -> str:
    temp_entry = Entry(entry_type="", key="", fields=fields)
    last = clean_key_part(first_author_last_name(temp_entry))
    year = extract_year(fields) or "0000"
    keyword = title_keyword(fields.get("title", ""))
    return f"{last}{year}{keyword}"


def parse_ids_field(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[,\s;]+", value)
    return [part for part in parts if part]


def collect_old_ids(entries: Iterable[Entry]) -> List[str]:
    ids: List[str] = []
    for entry in entries:
        ids.append(entry.key)
        ids.extend(parse_ids_field(entry.fields.get("ids", "")))
    return list(dict.fromkeys(ids))


def choose_base_entry(entries: List[Entry]) -> Entry:
    with_doi = [entry for entry in entries if entry.fields.get("doi")]
    if with_doi:
        return max(with_doi, key=entry_score)
    return max(entries, key=entry_score)


def merge_group(entries: List[Entry]) -> Tuple[Dict[str, str], str, Entry]:
    base = choose_base_entry(entries)
    merged_fields: Dict[str, str] = {}
    for field, value in base.fields.items():
        if field != "ids" and value and not is_corrupted_field_value(value):
            merged_fields[field] = value
    for entry in entries:
        for field, value in entry.fields.items():
            if field == "ids" or not value:
                continue
            if is_corrupted_field_value(value):
                continue
            if not merged_fields.get(field):
                merged_fields[field] = value
    base_key = build_base_key(merged_fields)
    return merged_fields, base_key, base


def suffix_letters(index: int) -> str:
    letters = []
    while index > 0:
        index -= 1
        letters.append(chr(ord("a") + (index % 26)))
        index //= 26
    return "".join(reversed(letters))


def should_drop_url(entry: Entry, fields: Dict[str, str]) -> bool:
    if not fields.get("doi") or not fields.get("url"):
        return False
    if is_arxiv(entry):
        return False
    return entry.entry_type in PUBLISHED_TYPES


def normalize_entry_for_output(entry: Entry) -> Entry:
    fields = dict(entry.fields)

    journaltitle = fields.get("journaltitle", "")
    journal = fields.pop("journal", "")
    if journaltitle:
        fields["journaltitle"] = journaltitle
    elif journal:
        fields["journaltitle"] = journal

    if "author" in fields and not is_corrupted_field_value(fields["author"]):
        formatted_authors = format_author_list(fields["author"])
        if formatted_authors:
            fields["author"] = formatted_authors
        else:
            fields.pop("author", None)

    if "title" in fields and not is_corrupted_field_value(fields["title"]):
        fields["title"] = protect_title_acronyms(fields["title"])

    if "pages" in fields:
        pages = format_pages(fields["pages"])
        if pages:
            fields["pages"] = pages
        else:
            fields.pop("pages", None)

    archiveprefix = fields.get("archiveprefix", "")
    if "eprinttype" not in fields and archiveprefix:
        eprint_type = normalize_eprint_type(archiveprefix)
        if eprint_type:
            fields["eprinttype"] = eprint_type
    if "eprintclass" not in fields and fields.get("primaryclass"):
        fields["eprintclass"] = fields["primaryclass"]
    if "eprint" in fields and "eprinttype" not in fields and is_arxiv(entry):
        fields["eprinttype"] = "arxiv"

    fields.pop("archiveprefix", None)
    fields.pop("primaryclass", None)

    for drop_field in ("editor", "address", "location"):
        fields.pop(drop_field, None)

    if should_drop_url(entry, fields):
        fields.pop("url", None)
        fields.pop("urldate", None)

    cleaned = {field: value for field, value in fields.items() if value}
    return Entry(entry_type=entry.entry_type, key=entry.key, fields=cleaned)


def format_entry(entry: Entry) -> str:
    normalized = normalize_entry_for_output(entry)
    lines = [f"@{normalized.entry_type}{{{normalized.key},"]
    seen = set()
    for field in FIELD_ORDER:
        value = latexify_value(field, normalized.fields.get(field, ""))
        if value:
            lines.append(f"  {field} = {{{value}}},")
            seen.add(field)
    for field in sorted(normalized.fields):
        if field in seen:
            continue
        value = latexify_value(field, normalized.fields.get(field, ""))
        if value:
            lines.append(f"  {field} = {{{value}}},")
    lines.append("}")
    return "\n".join(lines)


def entry_groups(entries: List[Entry]) -> List[List[int]]:
    signatures: Dict[str, List[int]] = {}
    for idx, entry in enumerate(entries):
        for sig in entry_signatures(entry):
            signatures.setdefault(sig, []).append(idx)

    uf = UnionFind(len(entries))
    for idxs in signatures.values():
        if len(idxs) < 2:
            continue
        first = idxs[0]
        for idx in idxs[1:]:
            uf.union(first, idx)

    clusters: Dict[int, List[int]] = {}
    for idx in range(len(entries)):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)
    return list(clusters.values())


def analyze_cluster(
    indices: List[int], entries: List[Entry]
) -> Tuple[
    List[List[int]],
    List[Tuple[str, str, Dict[str, Tuple[str, str]]]],
    List[Tuple[str, str, Dict[str, Tuple[str, str]], str]],
    List[int],
]:
    pair_info: Dict[Tuple[int, int], Tuple[bool, Dict[str, Tuple[str, str]], str]] = {}
    unmerged: List[Tuple[str, str, Dict[str, Tuple[str, str]]]] = []
    merged_diff: List[Tuple[str, str, Dict[str, Tuple[str, str]], str]] = []
    hard_conflict_indices: set[int] = set()

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            idx_a = indices[i]
            idx_b = indices[j]
            mergeable, conflicts, reason = compare_entries(entries[idx_a], entries[idx_b])
            pair_info[(idx_a, idx_b)] = (mergeable, conflicts, reason)
            if conflicts:
                key_a = entries[idx_a].key
                key_b = entries[idx_b].key
                if mergeable and reason:
                    merged_diff.append((key_a, key_b, conflicts, reason))
                elif not mergeable:
                    unmerged.append((key_a, key_b, conflicts))
                    hard_conflict_indices.update([idx_a, idx_b])

    sorted_indices = sorted(indices, key=lambda idx: entry_score(entries[idx]), reverse=True)
    groups: List[List[int]] = []
    for idx in sorted_indices:
        placed = False
        for group in groups:
            if all(
                pair_info.get(tuple(sorted((idx, other_idx))), (True, {}, ""))[0]
                for other_idx in group
            ):
                group.append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])
    return groups, unmerged, merged_diff, sorted(hard_conflict_indices)


def merge_entries(
    entries: List[Entry],
) -> Tuple[List[Entry], List[str], List[List[int]]]:
    merged_items: List[MergedItem] = []
    unmerged_conflicts: List[Tuple[str, str, Dict[str, Tuple[str, str]]]] = []
    merged_differences: List[Tuple[str, str, Dict[str, Tuple[str, str]], str]] = []
    conflict_clusters: List[List[int]] = []

    for cluster in entry_groups(entries):
        if len(cluster) == 1:
            entry = entries[cluster[0]]
            merged_fields, base_key, base = merge_group([entry])
            old_ids = collect_old_ids([entry])
            merged_items.append(
                MergedItem(
                    entry=Entry(entry_type=base.entry_type, key=base_key, fields=merged_fields),
                    old_ids=old_ids,
                )
            )
            continue

        groups, unmerged, merged_diff, hard_conflict_indices = analyze_cluster(cluster, entries)
        unmerged_conflicts.extend(unmerged)
        merged_differences.extend(merged_diff)
        if unmerged or merged_diff:
            conflict_clusters.append(sorted(cluster))

        for group in groups:
            group_entries = [entries[idx] for idx in group]
            merged_fields, base_key, base = merge_group(group_entries)
            old_ids = collect_old_ids(group_entries)
            merged_items.append(
                MergedItem(
                    entry=Entry(entry_type=base.entry_type, key=base_key, fields=merged_fields),
                    old_ids=old_ids,
                )
            )

    assign_unique_keys(merged_items)
    merged_entries = attach_ids(merged_items)
    report_lines = build_report(
        unmerged_conflicts, merged_differences, len(entries), len(merged_entries)
    )
    return merged_entries, report_lines, conflict_clusters


def assign_unique_keys(items: List[MergedItem]) -> None:
    counts: Dict[str, int] = {}
    for item in items:
        base = item.entry.key
        counts.setdefault(base, 0)
        counts[base] += 1
        if counts[base] == 1:
            continue
        item.entry.key = f"{base}{suffix_letters(counts[base] - 1)}"


def attach_ids(items: List[MergedItem]) -> List[Entry]:
    result: List[Entry] = []
    for item in items:
        ids = [entry_id for entry_id in item.old_ids if entry_id and entry_id != item.entry.key]
        ids = list(dict.fromkeys(ids))
        if ids:
            item.entry.fields["ids"] = ", ".join(ids)
        else:
            item.entry.fields.pop("ids", None)
        result.append(item.entry)
    return result


def build_report(
    unmerged: List[Tuple[str, str, Dict[str, Tuple[str, str]]]],
    merged_diff: List[Tuple[str, str, Dict[str, Tuple[str, str]], str]],
    original_count: int,
    merged_count: int,
) -> List[str]:
    lines = []
    lines.append(f"Input entries: {original_count}")
    lines.append(f"Output entries: {merged_count}")
    if unmerged:
        lines.append("")
        lines.append("UNMERGED_CONFLICTS")
        for key_a, key_b, conflicts in unmerged:
            lines.append(f"{key_a} <-> {key_b}")
            for field, values in conflicts.items():
                left, right = values
                lines.append(f"  {field}: {shorten_value(left)} || {shorten_value(right)}")
    if merged_diff:
        lines.append("")
        lines.append("MERGED_DIFFERENCES")
        for key_a, key_b, conflicts, reason in merged_diff:
            reason_label = reason.replace("_", " ").upper()
            lines.append(f"{key_a} <-> {key_b} ({reason_label})")
            for field, values in conflicts.items():
                left, right = values
                lines.append(f"  {field}: {shorten_value(left)} || {shorten_value(right)}")
    return lines


def shorten_value(value: str, limit: int = 160) -> str:
    text = value.replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def write_bibtex(path: str, entries: List[Entry], passthrough: Optional[List[str]] = None) -> None:
    blocks = []
    if passthrough:
        blocks.extend(passthrough)
    for entry in sorted(entries, key=lambda item: item.key.lower()):
        blocks.append(format_entry(entry))
    content = "\n\n".join(blocks).strip() + "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def prompt_path(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value


def ensure_bib_extension(path: str) -> str:
    if path.lower().endswith(".bib"):
        return path
    return f"{path}.bib"


def resolve_existing_bib_path(initial: Optional[str]) -> str:
    if initial:
        candidate = ensure_bib_extension(initial.strip())
        if os.path.isfile(candidate):
            return candidate
        print(f"Not found: {candidate}")
        if not sys.stdin.isatty():
            raise SystemExit(1)
    while True:
        value = prompt_path("Input .bib path (extension optional): ")
        candidate = ensure_bib_extension(value)
        if os.path.isfile(candidate):
            return candidate
        print(f"Not found: {candidate}")


def derive_default_path(input_bib: str, suffix: str) -> str:
    base = os.path.splitext(os.path.basename(input_bib))[0]
    dir_path = os.path.dirname(input_bib) or "."
    return os.path.join(dir_path, f"{base}{suffix}")


def default_output_path(input_bib: str) -> str:
    dir_path = os.path.dirname(input_bib) or "."
    return os.path.join(dir_path, "output.bib")


def resolve_corrections_path(cli_value: Optional[str], input_bib: str) -> Optional[str]:
    if cli_value:
        return cli_value if os.path.isfile(cli_value) else None
    candidates = [
        os.path.join(os.path.dirname(input_bib) or ".", "corrected.bib"),
        os.path.join(".", "corrected.bib"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, idx: int) -> int:
        if self.parent[idx] != idx:
            self.parent[idx] = self.find(self.parent[idx])
        return self.parent[idx]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def apply_corrections(entries: List[Entry], corrections: List[Entry]) -> List[str]:
    key_index: Dict[str, List[int]] = {}
    for idx, entry in enumerate(entries):
        key_index.setdefault(entry.key.lower(), []).append(idx)

    applied = 0
    unmatched: List[str] = []
    for correction in corrections:
        targets: set[int] = set()
        corr_key = correction.key.lower()
        targets.update(key_index.get(corr_key, []))
        for old_id in parse_ids_field(correction.fields.get("ids", "")):
            targets.update(key_index.get(old_id.lower(), []))
        if not targets:
            unmatched.append(correction.key)
            continue
        new_fields = {key: value for key, value in correction.fields.items() if key != "ids"}
        for idx in targets:
            entries[idx].entry_type = correction.entry_type
            entries[idx].fields = dict(new_fields)
            applied += 1

    report = [f"Corrections applied to {applied} entries."]
    if unmatched:
        report.append("Unmatched correction keys:")
        report.extend(unmatched)
    return report


def entry_fingerprint(entry: Entry) -> str:
    parts = [entry.entry_type.lower(), entry.key.lower()]
    for field in sorted(entry.fields):
        value = entry.fields.get(field, "")
        parts.append(f"{field}={normalize_field_value(field, value)}")
    return "|".join(parts)


def write_conflicts_bib(
    path: str,
    clusters: List[List[int]],
    entries: List[Entry],
) -> None:
    blocks: List[str] = []
    for idx, cluster in enumerate(clusters, start=1):
        keys = ", ".join(entries[item].key for item in cluster)
        blocks.append(f"@comment{{CONFLICT_GROUP_{idx}: {keys}}}")

        seen = set()
        for item in cluster:
            entry = entries[item]
            fingerprint = entry_fingerprint(entry)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            blocks.append(format_entry(entry))

    content = "\n\n".join(blocks).strip()
    if content:
        content += "\n"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)
