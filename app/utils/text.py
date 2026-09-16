"""Lossless source offsets and small, dependency-free lexical units.

Offsets are Python Unicode character offsets, not UTF-8 bytes or audio times.
Whitespace is not a token, but is never normalized in the original source.
This is not a language tokenizer; Han/kana characters permit incremental LCP.
"""

from dataclasses import dataclass
import unicodedata


@dataclass(frozen=True)
class TextToken:
    text: str
    start: int
    end: int


def _cjk(char):
    value = ord(char)
    return (0x3400 <= value <= 0x9FFF or 0x20000 <= value <= 0x323AF
            or 0x3040 <= value <= 0x30FF or 0x31F0 <= value <= 0x31FF)


def _cluster_end(text, start):
    end = start + 1
    while end < len(text):
        char = text[end]
        value = ord(char)
        if (unicodedata.combining(char) or unicodedata.category(char).startswith("M")
                or 0xFE00 <= value <= 0xFE0F or 0x1F3FB <= value <= 0x1F3FF
                or 0xE0100 <= value <= 0xE01EF):
            end += 1
        elif char == "\u200d" and end + 1 < len(text):
            end += 2
        else:
            break
    return end


def tokenize(text):
    """Return exact source spans; preserve combining/ZWJ sequences as units."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    tokens = []
    position = 0
    while position < len(text):
        char = text[position]
        if char.isspace():
            position += 1
            continue
        start = position
        position = _cluster_end(text, position)
        if (char.isalnum() or char == "_") and not _cjk(char):
            number = char.isdecimal()
            while position < len(text):
                following = text[position]
                if (following.isalnum() or following == "_") and not _cjk(following):
                    number = number and following.isdecimal()
                    position = _cluster_end(text, position)
                elif (following in "'’" and position + 1 < len(text)
                      and text[position + 1].isalpha() and not _cjk(text[position + 1])):
                    number = False
                    position = _cluster_end(text, position + 1)
                elif (number and following in ".," and position + 1 < len(text)
                      and text[position + 1].isdecimal()):
                    position = _cluster_end(text, position + 1)
                else:
                    break
        tokens.append(TextToken(text[start:position], start, position))
    return tuple(tokens)


def common_prefix_length(sequences):
    """Length shared by every sequence; an empty collection has no prefix."""
    if not sequences:
        return 0
    shortest = min(map(len, sequences))
    for index in range(shortest):
        if any(sequence[index] != sequences[0][index] for sequence in sequences[1:]):
            return index
    return shortest


def token_char_end(tokens, count):
    if type(count) is not int or not 0 <= count <= len(tokens):
        raise ValueError("token count is outside the source")
    return tokens[count - 1].end if count else 0
