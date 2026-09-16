"""Conservative EN/KO meaning-unit heuristics with exact source offsets.

This is deliberately not a grammar parser or an LLM completeness guarantee.
Unknown constructions wait; elapsed time, ASR finality and pauses are never
inputs to this gate. Limits cannot cut a sentence into a forced fragment.
"""
from dataclasses import dataclass
import re

from app.utils.text import tokenize


@dataclass(frozen=True)
class SegmentCandidate:
    start: int
    end: int
    text: str
    complete: bool
    reason: str
    token_count: int
    strong_boundary: bool


_END = frozenset(".!?。！？")
_CLOSING = frozenset('"”’»)]}）】')
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e", "g", "i"}
_DANGLING = {
    "a", "an", "the", "and", "or", "but", "because", "if", "although", "unless",
    "whether", "while", "when", "than", "that", "which", "whose", "to", "of", "for",
    "with", "without", "from", "into", "onto", "by", "at", "on", "in", "as", "per",
    "about", "approximately", "around", "between", "over", "under", "through",
    "is", "are", "was", "were", "am", "be", "been", "being", "have", "has", "had",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "not", "n't", "very", "more", "less", "most", "least", "too", "quite", "just",
    "my", "your", "his", "her", "its", "our", "their", "this", "these", "those", "please", "then",
}
_AUX = {"am", "is", "are", "was", "were", "be", "have", "has", "had", "do", "does",
        "did", "will", "would", "shall", "should", "can", "could", "may", "might", "must",
        "i'm", "it's", "that's", "you're", "we're", "they're", "he's", "she's",
        "isn't", "aren't", "wasn't", "weren't", "don't", "doesn't", "didn't", "can't", "won't"}
_WH = {"who", "what", "where", "when", "why", "how", "which"}
_VERBS = {
    "arrive", "arrives", "arrived", "work", "works", "worked", "fail", "fails", "failed",
    "go", "goes", "went", "come", "comes", "came", "leave", "leaves", "left", "stop", "stops",
    "eat", "eats", "ate", "drink", "drinks", "drank", "sleep", "sleeps", "slept",
    "want", "wants", "wanted", "need", "needs", "needed", "like", "likes", "liked",
    "love", "loves", "loved", "know", "knows", "knew", "think", "thinks", "thought",
    "say", "says", "said", "mean", "means", "meant", "believe", "believes", "believed",
    "send", "sends", "sent", "buy", "buys", "bought", "sell", "sells", "sold",
    "cost", "costs", "weigh", "weighs", "weighed", "take", "takes", "took",
    "weighing", "wait", "waits", "waited", "see", "sees", "saw", "hear", "hears", "heard",
    "agree", "agrees", "agreed", "happen", "happens", "happened", "matter", "matters",
    "open", "opens", "opened", "close", "closes", "closed", "start", "starts", "started",
    "finish", "finishes", "finished", "move", "moves", "moved", "run", "runs", "ran",
    "keep", "keeps", "kept", "make", "makes", "made", "get", "gets", "got",
    "live", "lives", "lived", "look", "looks", "looked", "seem", "seems", "seemed",
    "win", "wins", "won", "crash", "crashes", "crashed", "respond", "responds", "responded",
}
_NEEDS_OBJECT = {"want", "wants", "wanted", "need", "needs", "needed", "send", "sends", "sent",
                 "buy", "buys", "bought", "sell", "sells", "sold", "take", "takes", "took",
                 "make", "makes", "made", "get", "gets", "got", "keep", "keeps", "kept",
                 "say", "says", "said", "mean", "means", "meant", "think", "thinks", "thought",
                 "believe", "believes", "believed", "seem", "seems", "seemed", "look", "looks", "looked"}
_NEEDS_OBJECT.update({"cost", "costs", "weigh", "weighs", "weighed"})
_COMMANDS = {"stop", "go", "come", "wait", "help", "listen", "look", "run", "leave", "continue",
             "proceed", "sit", "stand", "stay", "relax", "open", "close", "send", "bring",
             "take", "give", "turn", "put", "hold", "read", "write", "try", "check", "call",
             "remember", "forget", "keep", "start", "finish", "move", "speak", "repeat"}
_TRANSITIVE_COMMANDS = {"open", "close", "send", "bring", "take", "give", "turn", "put", "hold",
                         "read", "write", "check", "call", "remember", "forget", "keep", "repeat"}
_NUMBERS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion half quarter dozen".split())
_UNITS = set("kg g kilograms kilogram grams gram pounds pound lbs lb tons ton tonnes tonne meters meter metres metre cm mm km miles mile feet foot inches inch minutes minute seconds second hours hour days day weeks week months month years year dollars dollar euros euro won yen percent percentage degrees degree celsius fahrenheit liters liter litres litre ml people persons items books tickets apples boxes packages".split())
_SUBORDINATE = {"if", "because", "although", "unless", "whether", "while", "since", "whenever", "whereas"}
_REPORTING = {"think", "thinks", "thought", "believe", "believes", "believed", "say", "says", "said"}
_SUBJECTS = {"i", "you", "he", "she", "it", "we", "they", "this", "that", "these", "those"}
_DETERMINERS = {"a", "an", "the", "my", "your", "his", "her", "its", "our", "their"}
_OBJECT_LIST_VERBS = {"buy", "buys", "bought", "sell", "sells", "sold", "send", "sends", "sent",
                      "like", "likes", "liked", "love", "loves", "loved", "want", "wants", "wanted",
                      "need", "needs", "needed", "see", "sees", "saw", "hear", "hears", "heard"}


def _numeric(word):
    return word in _NUMBERS or bool(re.fullmatch(r"\d+(?:[.,]\d+)*", word))


def _question_answer(words, context):
    """Only a small answer form paired with the nearest explicit question."""
    previous = context.strip()
    if not previous.endswith(("?", "？")):
        return False
    question = re.split(r"[.!?。！？]\s*", previous[:-1])[-1].lower().strip()
    joined = " ".join(words)
    if joined in {"yes", "no", "yes please", "no thanks", "certainly", "of course"}:
        return bool(question.split() and question.split()[0] in _AUX)
    if question.startswith("how many"):
        return bool(words) and all(_numeric(word) or word in _UNITS for word in words)
    if question.startswith(("how long", "how much", "how far", "how heavy")):
        return len(words) >= 2 and any(_numeric(word) for word in words) and words[-1] in _UNITS
    return False


def _english(text, context, punctuated, _depth=0):
    if _depth >= 8:
        return False, "complex_clause", False
    word_tokens = [token for token in tokenize(text) if any(char.isalnum() for char in token.text)]
    words = [token.text.casefold() for token in word_tokens]
    if not words:
        return False, "empty_source", False
    joined = " ".join(words)
    if _question_answer(words, context):
        return True, "question_answer", True
    if joined in {"thank you", "thanks", "hello", "goodbye", "good morning", "good evening", "good night", "sorry", "excuse me"}:
        return True, "complete_expression", True
    if words[-1] in {"i", "he", "she", "we", "they"}:
        return False, "dangling_clause_subject", False
    if words[-1] in _DANGLING:
        return False, "dangling_english_tail", False
    question_mark = punctuated and text.rstrip().rstrip('"”’»)]}').endswith(("?", "？"))
    if question_mark:
        if len(words) == 1 and words[0] in {"ready", "really", "why", "where", "who", "what"}:
            return True, "complete_question", True
        if words[0] in _WH:
            lexical = any(word in _VERBS for word in words[1:])
            copular = len(words) >= 3 and words[1] in {"is", "are", "was", "were"}
            if lexical or copular:
                return True, "complete_question", True
            return False, "incomplete_question", False
        if words[0] in _AUX and len(words) >= 3:
            if len(words) == 3 and words[1] in _DETERMINERS:
                return False, "incomplete_question", False
            return True, "complete_question", True
    if words[-1] in _NEEDS_OBJECT:
        return False, "dangling_english_tail", False
    if _numeric(words[-1]):
        # A bare amount is not evidence that its noun, unit or range is done.
        count_predicate = re.search(r"\b(?:count|number|answer|result|score|age)\s+(?:is|was|equals)\b", joined)
        if not count_predicate and not re.match(r"^(?:i am|i'm|he is|she is) \w+$", joined):
            return False, "dangling_amount", False
    if words[0] in _SUBORDINATE or (words[0] in {"when", "after", "before", "once", "as"}
                                     and (len(words) < 2 or words[1] not in _AUX)):
        # Require an independently complete clause after a real separator.
        pieces = re.split(r"[,;]", text, maxsplit=1)
        if len(pieces) != 2 or not _english(pieces[1], "", punctuated, _depth + 1)[0]:
            return False, "dependent_clause", False
        return True, "complete_conditional", punctuated
    if words[0] == "that" and (len(words) < 2 or words[1] not in _AUX):
        return False, "dependent_clause", False
    command = words[1:] if words[0] == "please" else words
    if command[:2] == ["do", "not"]:
        command = command[2:]
    elif command and command[0] == "don't":
        command = command[1:]
    if command and command[0] in _COMMANDS:
        if len(command) == 1 and command[0] in _TRANSITIVE_COMMANDS:
            return False, "missing_command_object", False
        return True, "complete_command", True
    if words[0] in {"let's", "lets"} and len(words) > 1:
        return True, "complete_command", True
    # A finite predicate needs a subject and some complement when transitive.
    predicate = next((index for index, word in enumerate(words)
                      if word in _AUX or word in _VERBS or (len(word) > 4 and word.endswith("ed"))), None)
    if (predicate is None or (predicate == 0 and "'" not in words[0])
            or (predicate == 1 and words[0] in {"a", "an", "the", "this", "that", "these", "those"})):
        return False, "no_independent_predicate", False
    if words[-1] in _AUX:
        return False, "dangling_english_tail", False
    reporting = next((index for index, word in enumerate(words) if word in _REPORTING), None)
    if reporting is not None and reporting + 1 < len(words):
        tail = words[reporting + 1:]
        # Topic prepositions and explicit short quoted answers are complete.
        topic = tail[0] in {"about", "of", "in"}
        direct_answer = " ".join(tail) in {"so", "yes", "no", "hello", "goodbye", "thank you", "thanks"}
        if not topic and not direct_answer:
            offset = reporting + 1 if tail[0] == "that" else reporting
            complement = text[word_tokens[offset].end:]
            if not _english(complement, "", punctuated, _depth + 1)[0]:
                return False, "incomplete_complement", False
    conjunction = next((index for index, word in enumerate(words)
                        if index > predicate and word in {"and", "but", "or", "so"}), None)
    if conjunction is not None and conjunction + 1 < len(words):
        tail = words[conjunction + 1:]
        # A new subject begins another clause; noun lists may share an object.
        object_list = words[predicate] in _OBJECT_LIST_VERBS and words[conjunction] in {"and", "or"}
        new_subject = tail[0] in _SUBJECTS or (tail[0] in _DETERMINERS and not object_list)
        if new_subject and not _english(text[word_tokens[conjunction].end:], "", punctuated, _depth + 1)[0]:
            return False, "incomplete_coordinated_clause", False
    return True, "complete_clause", punctuated


def _korean(text, context, punctuated):
    body = re.sub(r'[\s.!?。！？"”’»\])}]+$', "", text)
    last = body.split()[-1] if body.split() else ""
    if re.search(r"(?:지만|는데|은데|ㄴ데|으면|다면|라면|면서|으며|거나|어서|아서|니까|므로|려고|도록|하기|하는|되는|아니라|그리고|그래서|때문에|위해|대해|관해|보다|약|대략)$", last):
        return False, "dependent_korean_clause", False
    if re.search(r"(?:\d+(?:[.,]\d+)*|스물|서른|마흔|열|한|두|세|네|다섯|여섯|일곱|여덟|아홉|십|백|천|만)$", last):
        return False, "dangling_amount", False
    if body in {"안녕하세요", "감사합니다", "고맙습니다", "안녕", "죄송합니다", "멈춰", "멈추세요", "기다려", "도와주세요", "가세요"}:
        return True, "complete_expression", True
    if body in {"네", "예", "아니요", "아니오"} and context.strip().endswith(("?", "？")):
        return True, "question_answer", True
    if re.search(r"(?:습니다|습니까|입니다|입니까|합니다|합니까|됩니다|됩니까|세요|십시오|네요|군요|어요|아요|여요|해요|예요|이에요|지요|죠|까요|나요|가요|한다|된다|있다|없다|했다|였다|이다|좋다|싫다|싶다|같다|간다|온다|잔다|왔다|갔다|었다|았다|는다)$", last):
        return True, "complete_korean_ending", True
    return False, "no_complete_korean_ending", False


def _complete(text, context, language):
    stripped = text.strip()
    if not stripped or stripped.endswith((",", ":", ";", "-", "—", "/", "…")):
        return False, "open_boundary", False
    if stripped.endswith("..."):
        return False, "ellipsis", False
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}"), ("（", "）")):
        if stripped.count(opening) != stripped.count(closing):
            return False, "unclosed_source", False
    punctuated = stripped.rstrip('"”’»)]}）】').endswith(tuple(_END))
    if language and language.lower().startswith("ko") or re.search(r"[가-힣]", stripped):
        return _korean(stripped, context, punctuated)
    if re.search(r"[\u3400-\u9fff\u3040-\u30ff]", stripped) and not re.search(r"[A-Za-z]", stripped):
        return (True, "cjk_sentence_boundary", True) if punctuated else (False, "unclosed_cjk_sentence", False)
    return _english(stripped, context, punctuated)


def segment_prefix(text, stable_char_end, *, start=0, context="", language=None, max_tokens=24,
                   allow_oversize_complete=False):
    """Return the longest complete prefix within the token budget, or None.

    Leading/inter-sentence whitespace belongs to the returned original slice.
    Never skip an incomplete first clause to select a later, easier sentence.
    Opting into oversized complete sentences makes max_tokens a soft boundary
    preference only for the first strong, punctuated sentence. The caller must
    still enforce its source character limit before invoking this function.
    """
    if not isinstance(text, str) or not 0 <= start <= stable_char_end <= len(text) or max_tokens < 1:
        raise ValueError("Invalid source segment limits")
    tokens = tokenize(text)
    boundaries = {0, len(text), *(token.end for token in tokens)}
    if stable_char_end not in boundaries or start not in boundaries:
        raise ValueError("Segments must use exact source token boundaries")
    available = [token for token in tokens if token.start >= start and token.end <= stable_char_end]
    if not available:
        return None
    cursor = start
    selected = None
    consumed = 0
    index = 0
    while index < len(available):
        token = available[index]
        endpoint = token.text in _END
        if token.text == "." and index and available[index - 1].text.casefold() in _ABBREVIATIONS:
            endpoint = False
        if endpoint:
            end_index = index
            while end_index + 1 < len(available) and available[end_index + 1].text in _END | _CLOSING:
                end_index += 1
            end = available[end_index].end
            complete, reason, strong = _complete(text[cursor:end], context, language)
            count = end_index + 1
            if not complete:
                return selected
            if count > max_tokens:
                if selected is None and allow_oversize_complete and strong:
                    return SegmentCandidate(start, end, text[start:end], True, reason, count, strong)
                return selected
            selected = SegmentCandidate(start, end, text[start:end], True, reason, count, strong)
            context = text[cursor:end]
            cursor, consumed, index = end, count, end_index + 1
        else:
            index += 1
    if consumed < len(available) and len(available) <= max_tokens:
        end = available[-1].end
        complete, reason, strong = _complete(text[cursor:end], context, language)
        if complete:
            selected = SegmentCandidate(start, end, text[start:end], True, reason, len(available), strong)
    return selected


class SemanticSegmenter:
    """Replaceable EN/KO gateway adapter; direct helpers also demo CJK spans.

    Punctuation-only CJK detection is not advertised as meaning validation.
    A different language needs a dedicated segmenter, not an English fallback.
    """

    @staticmethod
    def supports(language):
        return isinstance(language, str) and language.lower().replace("_", "-").split("-", 1)[0] in {"en", "ko"}

    def select(self, text, stable_char_end, *, start=0, context="", language=None,
               max_tokens=24, allow_oversize_complete=False):
        if not self.supports(language):
            raise ValueError("UNKNOWN_SOURCE_LANGUAGE")
        return segment_prefix(text, stable_char_end, start=start, context=context, language=language,
                              max_tokens=max_tokens, allow_oversize_complete=allow_oversize_complete)
