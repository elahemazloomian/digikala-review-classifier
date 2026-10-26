"""Persian text normalization for Digikala review bodies.

Every rule is backed by a count from notebook 01 (section 11 of the report).
Characters are written as Unicode escapes so the file stays pure ASCII and
cannot be damaged by an editor's encoding.
"""

import re

# Arabic letters that look like Persian ones but are different characters.
ARABIC_TO_PERSIAN = str.maketrans({
    "\u064A": "\u06CC",  # Arabic yeh  -> Persian yeh
    "\u0643": "\u06A9",  # Arabic kaf  -> Persian kaf
})

# Persian (U+06F0..) and Arabic-Indic (U+0660..) digits -> Latin digits.
PERSIAN_DIGITS = "\u06F0\u06F1\u06F2\u06F3\u06F4\u06F5\u06F6\u06F7\u06F8\u06F9"
ARABIC_DIGITS = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"
LATIN_DIGITS = "0123456789"
TO_LATIN_DIGITS = str.maketrans(PERSIAN_DIGITS + ARABIC_DIGITS, LATIN_DIGITS * 2)

# Invisible characters, except the half-space U+200C which is valid Persian.
INVISIBLE = re.compile("[\u200b\u200d\u200e\u200f\ufeff]")
URL = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE = re.compile(r"\s+")


def normalize_text(text):
    """Return the normalized version of one review body."""
    if text is None:
        return ""
    text = text.translate(ARABIC_TO_PERSIAN)
    text = text.translate(TO_LATIN_DIGITS)
    text = INVISIBLE.sub("", text)
    text = URL.sub(" ", text)
    text = WHITESPACE.sub(" ", text)
    return text.strip()