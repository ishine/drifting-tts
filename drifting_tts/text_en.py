"""English text frontend: normalisation and a character vocabulary (``language: en``).

English spelling is far from phonemic, but with thousands of hours of speech the text encoder learns the letter-to-
sound mapping itself, as character-input TTS models do, so characters are the input symbols (no G2P dependency). The
alphabet is the 26 letters, the apostrophe (``don't``, ``we'll``: it changes the pronunciation) and the Turkish
frontend's punctuation; ``<pad>`` and ``<blank>`` keep their ids, so ``PAD_ID`` / ``BLANK_ID`` are shared.

Normalisation lower-cases and writes out what is not spelled as it is read: numbers (cardinals, decimals, years,
decades, ordinals), money, percentages, clock times, common abbreviations and all-caps acronyms (letter by letter).
"""

from __future__ import annotations

import re
import unicodedata

from .text import BLANK, PAD, PUNCTUATION, group_sentences, split_sentences  # noqa: F401 (re-exported)

LETTERS = "abcdefghijklmnopqrstuvwxyz'"
SYMBOLS = [PAD, BLANK, *LETTERS, *PUNCTUATION]
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
PAD_ID, BLANK_ID = SYMBOL_TO_ID[PAD], SYMBOL_TO_ID[BLANK]

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (10**3, "thousand")]
_ORDINAL_WORDS = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
                  "nine": "ninth", "twelve": "twelfth"}
_LETTER_NAMES = dict(zip("abcdefghijklmnopqrstuvwxyz",
                         "ay bee see dee ee eff gee aitch eye jay kay el em en oh pee cue ar ess tee you vee "
                         "double-you ex why zee".split()))
# all-caps words spelled letter by letter: those without a vowel (TV, HR, NFL) and these; other all-caps words are
# read as words (emphasis: "I was SO tired"; NASA, NATO)
_SPELLED = {"USA", "UK", "EU", "UN", "AI", "CEO", "CFO", "CIA", "FBI", "DNA", "UFO", "IOU", "ATM", "BMW", "HIV", "IQ",
            "ID", "UI", "API", "URL", "USB", "GPU", "CPU", "OS", "EA", "ESPN", "MIT", "UCLA", "ICU", "IRS", "ER",
            "PE", "AA", "BA", "MA", "VIP", "OJ"}
_INTERJECTIONS = {"HMM", "HM", "MM", "MMM", "SHH", "SH", "PSST", "BRR", "GRR", "ZZZ", "PFFT", "TSK", "NTH", "PST"}
_SPECIAL = {"OK": "okay", "Ok": "okay", "BBQ": "barbecue", "Mr": "mister", "Mrs": "missus", "Ms": "miz",
            "Dr": "doctor", "Jr": "junior", "Sr": "senior", "St": "saint", "Prof": "professor"}
_ABBREVIATIONS = {"vs": "versus", "etc": "et cetera", "approx": "approximately", "e.g": "for example",
                  "i.e": "that is"}
_CHAR_MAP = {
    "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'", "“": '"', "”": '"', "«": '"', "»": '"',
    "…": "...", "—": ", ", "–": "-", "−": "-", ";": ",",
    "&": " and ", "+": " plus ", "=": " equals ", "@": " at ", "#": " number ",
}
_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
             "€": ("euro", "euros", "cent", "cents")}
_NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?"
_SCALE_WORD = r"(?:thousand|million|billion|trillion|k|m|bn)\b"
_TIME = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?![\d:])(\s*(?:[ap]\.?m\b\.?))?", re.I)
_AMPM = re.compile(r"(?<=\d)\s*([ap])\.?m\b\.?", re.I)
_HOUR_WORD = r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|o'clock)"
_AMPM_WORDS = re.compile(rf"({_HOUR_WORD})\s+([AaPp])\.?[Mm]\b\.?")  # "three AM", "nine p.m."


def _below_thousand(n: int) -> list[str]:
    out = []
    if n >= 100:
        out += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        out.append(_TENS[n // 10])
        n %= 10
        if n:
            out.append(_ONES[n])
    elif n or not out:
        out.append(_ONES[n])
    return out


def number_to_words(n: int) -> str:
    """Cardinal number in words: 2024 -> "two thousand twenty four" (American: no "and")."""
    if n < 0:
        return "minus " + number_to_words(-n)
    if n >= 10**15:
        return " ".join(_ONES[int(d)] for d in str(n))
    out = []
    for value, name in _SCALES:
        if n >= value:
            out += _below_thousand(n // value) + [name]
            n %= value
    if n or not out:
        out += _below_thousand(n)
    return " ".join(out)


def ordinal_to_words(n: int) -> str:
    words = number_to_words(n).split()
    last = words[-1]
    if last in _ORDINAL_WORDS:
        words[-1] = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def year_to_words(n: int) -> str:
    """1999 -> "nineteen ninety nine", 1900 -> "nineteen hundred", 2008 -> "two thousand eight", 2024 -> "twenty twenty
    four"."""
    hi, lo = divmod(n, 100)
    if 2000 <= n < 2010 or n % 1000 == 0:
        return number_to_words(n)
    if lo == 0:
        return f"{number_to_words(hi)} hundred"
    return f"{number_to_words(hi)} {'oh ' if lo < 10 else ''}{number_to_words(lo)}"


def _plural(words: str) -> str:
    """ "nineteen ninety" -> "nineteen nineties" (decades)."""
    head, _, last = words.rpartition(" ")
    last = last[:-1] + "ies" if last.endswith("y") else last + ("es" if last.endswith("x") else "s")
    return f"{head} {last}".strip()


def _number(s: str, year: bool = True) -> str:
    s = s.replace(",", "")
    if "." in s:
        whole, frac = s.split(".", 1)
        return f"{number_to_words(int(whole))} point {' '.join(_ONES[int(d)] for d in frac)}"
    n = int(s)
    if year and 1100 <= n < 2100 and len(s) == 4:
        return year_to_words(n)
    if len(s) > 1 and s[0] == "0":  # 007, 0800: digit by digit
        return " ".join("oh" if d == "0" else _ONES[int(d)] for d in s)
    return number_to_words(n)


def _scale(word: str | None) -> str:
    return {"k": "thousand", "m": "million", "bn": "billion"}.get((word or "").lower(), (word or "").lower())


def _money(m: re.Match) -> str:
    sym, amount, scale = m.group(1), m.group(2).replace(",", ""), _scale(m.group(3))
    one, many, sub_one, sub_many = _CURRENCY[sym]
    if scale:
        return f" {_number(amount, year=False)} {scale} {many} "
    whole, _, cents = amount.partition(".")
    out = f"{number_to_words(int(whole))} {one if int(whole) == 1 else many}"
    if cents and int(cents[:2].ljust(2, "0")):
        c = int(cents[:2].ljust(2, "0"))
        out += f" {number_to_words(c)} {sub_one if c == 1 else sub_many}"
    return f" {out} "


def _stop(m: re.Match) -> str:
    """"." when the match swallowed the full stop that ends a sentence ("at 5 p.m. Then"), else ""."""
    return "." if m.group().endswith(".") and re.match(r"\s*($|[A-Z\"'])", m.string[m.end():]) else ""


def _time(m: re.Match) -> str:
    h, mm, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if h > 24 or mm > 59:
        return f"{number_to_words(h)} {number_to_words(mm)}"
    words = number_to_words(h)
    if mm:
        words += f" {'oh ' if mm < 10 else ''}{number_to_words(mm)}"
    elif not ampm:
        words += " o'clock"
    if ampm:
        words += " ay em" if ampm.strip().lower().startswith("a") else " pee em"
    return f" {words}{_stop(m)} "


def _caps(m: re.Match) -> str:
    """An all-caps word (or a known abbreviation without its period): spelled, special-cased or kept."""
    word = m.group()
    if word in _SPECIAL:
        return _SPECIAL[word]
    plural = len(word) > 2 and word.endswith("s") and word[:-1].isupper()  # "TVs", "CDs"
    letters = word[:-1] if plural else word
    if not letters.isupper() or len(letters) < 2 or letters in _INTERJECTIONS:
        return word
    if letters in _SPELLED or (len(letters) <= 5 and not re.search(r"[AEIOUY]", letters)):
        return f" {' '.join(_LETTER_NAMES[c.lower()] for c in letters)}{'s' if plural else ''} "
    return word


def normalize(text: str) -> str:
    """Normalise raw English text to the symbol alphabet (lower-case letters, apostrophe, space and ``.,!?``)."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))  # café -> cafe
    text = "".join(_CHAR_MAP.get(c, c) for c in text)
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Jr|Sr|St|Prof)\.(?=\s)", lambda m: _SPECIAL[m.group(1)], text)
    text = re.sub(r"\b(vs|etc|approx|e\.g|i\.e)\.", lambda m: f" {_ABBREVIATIONS[m.group(1).lower()]}{_stop(m)} ",
                  text, flags=re.I)
    text = re.sub(r"\b[Nn]o\.\s*(?=\d)", "number ", text)
    text = _TIME.sub(_time, text)
    text = _AMPM.sub(lambda m: f" {'ay' if m.group(1).lower() == 'a' else 'pee'} em{_stop(m)} ", text)
    text = _AMPM_WORDS.sub(lambda m: f"{m.group(1)} {'ay' if m.group(2).lower() == 'a' else 'pee'} em{_stop(m)} ",
                           text)
    text = text.replace(":", ",")
    text = re.sub(rf"([$£€])\s*({_NUMBER})(?:\s*({_SCALE_WORD}))?", _money, text, flags=re.I)
    text = re.sub(rf"({_NUMBER})\s*%", lambda m: f" {_number(m.group(1), year=False)} percent ", text)
    text = re.sub(r"'?(\d{2}|\d{4})'?s\b", lambda m: f" {_plural(_number(m.group(1)))} ", text)  # the 90s, 1980's
    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: f" {ordinal_to_words(int(m.group(1)))} ", text, flags=re.I)
    text = re.sub(r"(?<![\w.,])-(?=\d)", " minus ", text)
    text = re.sub(rf"({_NUMBER})\s*-\s*(?={_NUMBER})", r"\1 to ", text)  # 20-30
    text = re.sub(_NUMBER, lambda m: f" {_number(m.group())} ", text)
    text = re.sub(r"\b[A-Z][A-Za-z]*\b", _caps, text)
    text = text.lower()
    text = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", text)  # quote marks; keep don't, we'll, rock'n'roll
    text = text.replace('"', " ").replace("-", " ").replace("/", " ")
    text = "".join(c if c in SYMBOL_TO_ID else " " for c in text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)  # no space before punctuation
    text = re.sub(r",+", ",", text)
    text = re.sub(r",(?=[.!?])", "", text)
    text = re.sub(r"([.,!?])(?=[^\s.,!?])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[.,!? ]+", "", text)
    return text


def text_to_ids(text: str, intersperse_blank: bool = True, normalized: bool = False) -> list[int]:
    """Map text to symbol ids; optionally intersperse a blank token (Grad-TTS) for alignment."""
    if not normalized:
        text = normalize(text)
    ids = [SYMBOL_TO_ID[c] for c in text]
    if intersperse_blank:
        out = [BLANK_ID] * (2 * len(ids) + 1)
        out[1::2] = ids
        ids = out
    return ids


def ids_to_text(ids: list[int]) -> str:
    return "".join(SYMBOLS[i] for i in ids if i not in (PAD_ID, BLANK_ID))
