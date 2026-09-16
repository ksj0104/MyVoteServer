"""Source-only revision planning. Translated character positions are unrelated."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RevisionSegment:
    start: int
    end: int


@dataclass(frozen=True)
class RevisionPlan:
    rollback_segment_index: int
    rollback_char_end: int
    replacement_text: str
    changed: bool


def plan_revision(old_text, new_text, segments):
    """Keep whole unchanged segments; invalidate the first touched and all later.

    `segments` must be ordered source spans for this utterance. Appending to an
    existing word touches its segment too, while appending after a complete
    sentence leaves that segment intact. The caller decides whether an ASR
    final authorizes applying this plan; partial conflicts must merely wait.
    """
    from app.utils.text import tokenize

    previous = 0
    for segment in segments:
        if not 0 <= segment.start <= segment.end <= len(old_text) or segment.start < previous:
            raise ValueError("Revision source spans must be ordered and disjoint")
        previous = segment.end
    difference = 0
    for old, new in zip(old_text, new_text):
        if old != new:
            break
        difference += 1
    # A lexical extension is a correction even when its old characters survive.
    if difference == len(old_text) and len(new_text) > len(old_text):
        for token in tokenize(new_text):
            if token.start < difference < token.end:
                difference = token.start
                break
    index = len(segments)
    boundary = segments[-1].end if segments else 0
    if old_text != new_text:
        for position, segment in enumerate(segments):
            if difference < segment.end:
                index, boundary = position, segment.start
                break
    return RevisionPlan(index, boundary, new_text[boundary:], old_text != new_text)
