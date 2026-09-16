"""Match only relevant glossary entries; never rewrite model output mechanically."""


def matching_entries(source: str, entries: dict[str, str], *, budget: int = 2000) -> dict[str, str]:
    selected: dict[str, str] = {}
    folded_source = source.casefold()

    def latin_word(character: str) -> bool:
        return character.isascii() and (character.isalnum() or character == "_")

    def matches(key: str) -> bool:
        folded_key = key.casefold()
        start = folded_source.find(folded_key)
        while start >= 0:
            end = start + len(folded_key)
            left = (not latin_word(key[0]) or start == 0 or not latin_word(folded_source[start - 1]))
            right = (not latin_word(key[-1]) or end == len(folded_source) or not latin_word(folded_source[end]))
            if left and right:
                return True
            start = folded_source.find(folded_key, start + 1)
        return False

    for key, value in sorted(entries.items(), key=lambda pair: (-len(pair[0]), pair[0])):
        # Avoid compiling hundreds of regexes on the translation hot path.
        # Latin substrings are excluded, while Korean particles (CUDA를) may follow.
        cost = len((key + value).encode("utf-8"))
        if cost <= budget and matches(key):
            selected[key] = value
            budget -= cost
    return selected
