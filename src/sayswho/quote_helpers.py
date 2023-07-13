from collections import namedtuple
import regex as re
from itertools import zip_longest
from spacy.tokens import Doc, Span, Token
from typing import Iterable, Union, List, Literal
from .constants import (
    _reporting_verbs,
    _VERB_MODIFIER_DEPS,
    QUOTATION_MARK_PAIRS,
    ALL_QUOTES,
    BRACK_REGEX,
    DOUBLE_QUOTES,
    DOUBLE_QUOTES_NOSPACE_REGEX,
)
from spacy.symbols import VERB, PUNCT

DQTriple: tuple[list[Token], list[Token], Span] = namedtuple(
    "DQTriple", ["speaker", "cue", "content"]
)


def filter_cue_candidates(tok):
    return all([tok.pos == VERB, tok.lemma_ in _reporting_verbs])


def filter_speaker_candidates(ch, i, j):
    return all(
        [
            ch.pos != PUNCT,
            ((ch.i >= i and ch.i >= j) or (ch.i <= i and ch.i <= j)),
        ]
    )


def filter_quote_tokens(tok: Token, qtok_idx_pairs: List[tuple]) -> bool:
    return any(i <= tok.i <= j for i, j in qtok_idx_pairs)


def get_qtok_idx_pairs(doc: Union[Doc, Span]) -> List[tuple]:
    qtoks = [tok for tok in doc if tok.is_quote or (re.match(r"(\n)+", tok.text))]
    qtok_idx_pairs = [(-1, -1)]
    for n, q in enumerate(qtoks):
        if (
            not bool(q.whitespace_)
            and q.i not in [q_[1] for q_ in qtok_idx_pairs]
            and q.i > qtok_idx_pairs[-1][1]
        ):
            for q_ in qtoks[n + 1 :]:
                if (ord(q.text), ord(q_.text)) in QUOTATION_MARK_PAIRS:
                    qtok_idx_pairs.append((q.i, q_.i))
                    break
    return qtok_idx_pairs[1:]


def expand_noun(tok: Token) -> list[Token]:
    """Expand a noun token to include all associated conjunct and compound nouns."""
    tok_and_conjuncts = [tok] + list(tok.conjuncts)
    compounds = [
        child
        for tc in tok_and_conjuncts
        for child in tc.children
        if child.dep_ == "compound"
    ]
    return tok_and_conjuncts + compounds


def expand_verb(tok: Token) -> list[Token]:
    """Expand a verb token to include all associated auxiliary and negation tokens."""
    verb_modifiers = [
        child for child in tok.children if child.dep in _VERB_MODIFIER_DEPS
    ]
    return [tok] + verb_modifiers


def get_sent_idxs(span):
    indexes = [
        n
        for n, s in enumerate(span.doc.sents)
        if (s.start <= span.start <= s.end) or (s.start <= span.end <= s.end)
    ]
    return indexes[0], indexes[-1]


def line_break_window(span):
    """
    Finds the boundaries of the paragraph containing doc[i:j].
    """
    lb_tok_idxs = (
        [0]
        + [tok.i for tok in span.doc if re.match(r"\n", tok.text)]
        + [span.doc[-1].i]
    )
    for i_, j_ in zip_longest(lb_tok_idxs, lb_tok_idxs[1:]):
        if i_ <= span.start and j_ >= span.end:
            return (i_, j_)
    else:
        return (None, None)


def windower(span, method: Literal["overlap", "linebreaks"] = None):
    if method == "overlap":
        return [
            sent
            for sent in span.doc.sents
            if (sent.start < span.start < sent.end)
            or (sent.start < span.end < sent.end)
        ]
    else:
        i_sent, j_sent = get_sent_idxs(span)
        sents = (
            list(span.doc.sents)[i_sent - 1 : j_sent + 2]
            if i_sent > 0
            else list(span.doc.sents)[: j_sent + 2]
        )
        if method == "linebreaks":
            linebreaks = (
                [0]
                + [tok.i for tok in span.doc if re.match(r"\n", tok.text)]
                + [span.doc[-1].i]
            )
            linebreak_limits = [
                lb for lb in linebreaks if sents[0].start < lb <= span.end + 1
            ]
            if linebreak_limits:
                return [s for s in sents if s.end <= max(linebreak_limits)]
        return sents


def old_windower(span, lb_boundaries=False) -> Iterable:
    if lb_boundaries:
        i_, j_ = line_break_window(span)
        if i_ is not None and j_ is not None:
            return list(span.doc[i_ + 1 : j_ - 1].sents)
        else:
            return [span]

    else:
        i, j = span.start, span.end
        return [
            sent
            for sent in span.doc.sents
            # these boundary cases are a subtle bit of work...
            if (
                (sent.start < i and sent.end >= i - 1)
                or (sent.start <= j + 1 and sent.end > j)
            )
        ]


def para_quote_fixer(p, exp: bool = False):
    if not p:
        return
    p = p.strip()
    p = p.replace("''", '"')
    p = re.sub(r"(.{3,8}s\')(\s)", r"\1x\2", p)

    while re.search(DOUBLE_QUOTES_NOSPACE_REGEX, p):
        match = re.search(DOUBLE_QUOTES_NOSPACE_REGEX, p)
        if (
            len(re.findall(BRACK_REGEX.format(DOUBLE_QUOTES), p[: match.start()])) % 2
            != 0
        ):
            replacer = '" '
        else:
            replacer = ' "'
        p = p[: match.start()] + replacer + p[match.end() :]
    if (
        not (p[0] == "'" and p[-1] == "'")
        and p[0] in ALL_QUOTES
        and len(re.findall(BRACK_REGEX.format(DOUBLE_QUOTES), p[1:])) % 2 == 0
    ):
        p += '"'
    return p


def prep_text_for_quote_detection(t, para_char="\n", exp: bool = False):
    return para_char.join(
        [para_quote_fixer(p, exp=exp) for p in t.split(para_char) if p]
    )
