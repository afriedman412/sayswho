"""
This is just my version of the Textacy quote attributor. Will remove this if/when it gets implemented in Textacy!
"""

from . import constants
from spacy.tokens import Doc, Token, Span
from spacy.symbols import VERB, PUNCT
from operator import attrgetter
import regex as re
from typing import Literal, Iterable

def direct_quotations(doc: Doc):
    """
    
    """
    qtoks = [tok for tok in doc if tok.is_quote or (re.match(r"\n", tok.text))]
    qtok_idx_pairs = [(-1, -1)]
    for n, q in enumerate(qtoks):
        if (
            not bool(q.whitespace_)
            and q.i not in [q_[1] for q_ in qtok_idx_pairs]
            and q.i > qtok_idx_pairs[-1][1]
        ):
            for q_ in qtoks[n + 1 :]:
                if (ord(q.text), ord(q_.text)) in constants.QUOTATION_MARK_PAIRS:
                    qtok_idx_pairs.append((q.i, q_.i))
                    break
    qtok_idx_pairs = qtok_idx_pairs[1:]

    def filter_quote_tokens(tok):
        return any(qts_idx <= tok.i <= qte_idx for qts_idx, qte_idx in qtok_idx_pairs)

    for qtok_start_idx, qtok_end_idx in qtok_idx_pairs:
        content = doc[qtok_start_idx:qtok_end_idx]
        cue = None
        speaker = None

        if (
            len(content.text.split()) < constants.MIN_QUOTE_LENGTH
            # filter out titles of books and such, if possible
            or all(tok.is_title for tok in content if not (tok.is_punct or tok.is_stop))
        ):
            continue

        for window_sents in [
            windower(content, "overlap"),
            windower(content, "linebreaks"),
        ]:
            # get candidate cue verbs in window
            cue_candidates = [
                tok
                for sent in window_sents
                for tok in sent
                if tok.pos == VERB
                and tok.lemma_ in constants._reporting_verbs
                and not filter_quote_tokens(tok)
            ]
            cue_candidates = sorted(
                cue_candidates,
                key=lambda cc: min(
                    abs(cc.i - qtok_start_idx), abs(cc.i - qtok_end_idx)
                ),
            )
            for cue_cand in cue_candidates:
                if cue is not None:
                    break
                speaker_cands = [
                    speaker_cand
                    for speaker_cand in cue_cand.children
                    if speaker_cand.pos != PUNCT
                    and not filter_quote_tokens(speaker_cand)
                    and (
                        (speaker_cand.i >= qtok_end_idx)
                        or (speaker_cand.i <= qtok_start_idx)
                    )
                ]
                for speaker_cand in speaker_cands:
                    if speaker_cand.dep in constants._ACTIVE_SUBJ_DEPS:
                        cue = expand_verb(cue_cand)
                        speaker = expand_noun(speaker_cand)
                        break
            if content and cue and speaker:
                yield constants.DQTriple(
                    speaker=sorted(speaker, key=attrgetter("i")),
                    cue=sorted(cue, key=attrgetter("i")),
                    content=doc[qtok_start_idx : qtok_end_idx + 1],
                )
                break


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
        child for child in tok.children if child.dep in constants._VERB_MODIFIER_DEPS
    ]
    return [tok] + verb_modifiers


def windower(quote: Span, method: Literal["overlap", "linebreaks"]) -> Iterable[Span]:
    """
    Finds the range of sentences in which to look for quote attribution.

    3 ways:
    - "overlap": any sentences that overlap with the quote span
    - "linebreaks": overlap sentences +/- one sentence, without crossing linebreaks after the quote
    - None: overlap sentences +/- one sentence,

    Input:
        quote (Span) - quote to be attributed
        method (str) - how the sentence range will be determined

    Output:
        sents (list) - list of sentences
    """
    if method == "overlap":
        return [
            sent
            for sent in quote.doc.sents
            if (sent.start < quote.start < sent.end)
            or (sent.start < quote.end < sent.end)
        ]
    else:
        sent_indexes = [
            n
            for n, s in enumerate(quote.doc.sents)
            if (s.start <= quote.start <= s.end) or (s.start <= quote.end <= s.end)
        ]

        i_sent = sent_indexes[0] - 1 if sent_indexes[0] > 0 else 0
        j_sent = sent_indexes[-1] + 2
        sents = list(quote.doc.sents)[i_sent:j_sent]
        if method == "linebreaks":
            linebreaks = (
                [0]
                + [tok.i for tok in quote.doc if re.match(r"\n", tok.text)]
                + [quote.doc[-1].i]
            )
            linebreak_limits = [
                lb for lb in linebreaks if sents[0].start < lb <= quote.end + 1
            ]
            if linebreak_limits:
                return [s for s in sents if s.end <= max(linebreak_limits)]
        return sents


def prep_text_for_quote_detection(t: str, fix_plural_possessives: bool = True) -> str:
    """
    Sorts out some common issues that trip up the quote detector.
    Works best one paragraph at a time -- use prep_document_for_quote_detection for the whole doc.

    - replaces consecutive apostrophes with a double quote (no idea why this happens but it does)
    - adds spaces before or after double quotes that don't have them
    - if enabled, fixes plural possessives by adding an "x", because the hanging apostrophe can trigger quote detection.
    - adds a double quote to the end of paragraphs that are continuations of quotes and thus traditionally don't end with quotation marks

    Input:
        t (str) - text to be prepped, preferably one paragraph
        fix_plural_possessives (bool) - enables fix_plural_possessives

    Output:
        t (str) - text prepped for quote detection
    """
    if not t:
        return

    t = t.replace("''", '"')
    if fix_plural_possessives:
        t = re.sub(r"(.{3,8}s\')(\s)", r"\1x\2", t)
    while re.search(constants.DOUBLE_QUOTES_NOSPACE_REGEX, p):
        match = re.search(constants.DOUBLE_QUOTES_NOSPACE_REGEX, p)
        if (
            len(re.findall(constants.ANY_DOUBLE_QUOTE_REGEX, p[: match.start()])) % 2
            != 0
        ):
            replacer = '" '
        else:
            replacer = ' "'
        p = p[: match.start()] + replacer + p[match.end() :]
    if (
        not (p[0] == "'" and p[-1] == "'")
        and p[0] in constants.ALL_QUOTES
        and len(re.findall(constants.ANY_DOUBLE_QUOTE_REGEX, p[1:])) % 2 == 0
    ):
        p += '"'
    return p.strip()


def prep_document_for_quote_detection(t: str, para_char: str = "\n") -> str:
    """
    Splits text into paragraphs (on para_char), runs prep_text_for_quote_detection on all paragraphs, then reassembles with para_char.

    Input:
        t (str) - document to prep for quote detection
        para_char (str) - paragraph boundary in t

    Output:
        document prepped for quote detection
    """
    return para_char.join(
        [prep_text_for_quote_detection(t) for t in t.split(para_char) if t]
    )
