from .constants import (
    MIN_ENTITY_DIFF,
    MIN_SPEAKER_DIFF,
    Boundaries,
    _reporting_verbs,
    _VERB_MODIFIER_DEPS,
    QUOTATION_MARK_PAIRS,
    ALL_QUOTES,
    BRACK_REGEX,
    DOUBLE_QUOTES,
    DOUBLE_QUOTES_NOSPACE_REGEX,
)
import statistics
from itertools import zip_longest
from collections import namedtuple
import regex as re
from typing import Union, Literal, Tuple, Iterable, List
from rapidfuzz import fuzz
from spacy.tokens import Span, SpanGroup, Token, Doc
from spacy.symbols import VERB, PUNCT

DQTriple: tuple[list[Token], list[Token], Span] = namedtuple(
    "DQTriple", ["speaker", "cue", "content"]
)


def get_cluster_people_scores(
    cluster: SpanGroup, scorer: Literal["prat", "cos"] = "prat"
) -> Tuple[list, float]:
    """
    Calculates average similarity between any two PERSONS in the cluster.
    These scores are used to exclude "odd man out" cluster members.

    TODO: tweak cutoff value, or at least make it flexible

    Input:
        cluster (SpanGroup) - coref cluster
        scorer (str) - what score to use to determine similarity. can be 'prat' (partial ratio) or 'cos' (cosine similarity).

    Output:
        list(tuple) - index, span, average score for each span in the cluster
        cutoff (float) - minimum score for keeping cluster member (mean - 2stdev)
    """
    if scorer == "prat":
        score_func = lambda s1, s2: fuzz.partial_ratio(s1.text, s2.text)
    elif scorer == "cos":
        score_func = lambda s1, s2: s1.similarity(s2)

    # filter out non-persons
    cluster_ = [span for span in cluster if person_check(span)]

    all_scores = []
    for n, span in enumerate(cluster_):
        scores = [score_func(span, span_) for span_ in cluster_]
        if scores:
            all_scores.append((n, span, sum(scores) / len(scores)))
    if len(all_scores) > 1:
        cutoff = statistics.mean([c[-1] for c in all_scores]) - 2 * statistics.stdev(
            [c[-1] for c in all_scores]
        )
        return sorted(all_scores, key=lambda k: k[-1]), cutoff
    else:
        return [], None


def prune_cluster_people(cluster: SpanGroup, scorer="prat") -> list:
    """
    Removes outlier PERSONS from a cluster, based on provided score.
    TODO: SpanGroup instead of list?

    Input:
        cluster (SpanGroup) - a coref cluster

    Output:
        list - coref cluster with outlier PERSONS removed
    """
    scores, cutoff = get_cluster_people_scores(cluster, scorer=scorer)
    filtered = [c[1] for c in scores if c[-1] < cutoff]
    return [c for c in cluster if c not in filtered]


def clone_cluster(cluster: SpanGroup, destination_doc: Doc):
    """
    For copying a coref cluster to a different Doc.

    Necessary because I'm using multiple models (of different sizes) to do coreferencing and other NLP tasks.
    It's easier to consolidate the clusters than to combine the tasks into one model.
    """
    return SpanGroup(
        doc=destination_doc,
        spans=([destination_doc[span.start : span.end] for span in cluster]),
    )


def filter_duplicate_ents(ents) -> tuple:
    """
    Removes duplicate entities by text.
    """
    ent_bucket = []

    for e in ents:
        if e.text in [e.text for e in ent_bucket]:
            continue
        else:
            ent_bucket.append(e)

    return ent_bucket


def get_boundaries(t: Union[Token, Span, list, DQTriple]) -> Boundaries:
    """
    Convenience function that returns a Boundaries tuple with the start and the end character of t.

    Necessary because Token and Span have differently named attributes.

    Input:
        t (Token or Span) - spacy token or span object

    Output:
        Boundares(start, end) with start character and end character of t.
    """
    if isinstance(t, Token):
        return Boundaries(t.idx, t.idx + len(t))

    elif isinstance(t, Span):
        return Boundaries(t.start_char, t.end_char)

    elif isinstance(t, tuple):  # isinstance won't recognize DQTriple
        return Boundaries(
            get_boundaries(t.speaker[0]).start,
            get_boundaries(t.speaker[-1]).end,
        )

    else:
        raise TypeError("input needs to be Token, Span or DQTriple!")


def get_text(t: Union[Token, Span, list]) -> str:
    """
    Convenience function, because quote speakers are lists of tokens.
    """
    if isinstance(t, Token) or isinstance(t, Span):
        return t.text
    if isinstance(t, list):
        return " ".join([t_.text for t_ in t])


def span_contains(
    t1: Union[Span, Token, DQTriple], t2: Union[Span, Token, DQTriple]
) -> bool:
    """
    Does t1 contain t2 or v/v? Assumes both are from the same doc, or at least same index.

    Uses get_boundaries beacuse quote speakers are tokens but entities are spans.

    TODO: verify same doc

    Input:
        t1 and t2 - spacy spans or tokens

    Output:
        bool - whether either t1 contains the other t2

    """
    b1, b2 = tuple([get_boundaries(t) for t in [t1, t2]])

    if (b1.start <= b2.start and b1.end >= b2.end) or (
        b2.start <= b1.start and b2.end >= b1.end
    ):
        return True

    else:
        return False


def format_cluster(cluster):
    return list(set([c.text for c in cluster if c[0].pos_ != "PRON"]))


def pronoun_check(t: Span, doc: Doc = None):
    """
    Checks to see if t is a single pronoun.
    """
    if len(t) == 1:
        if doc:
            return doc[t[0].i].pos_ == "PRON"
        else:
            return t[0].pos_ == "PRON"
    return False


def person_check(span: Span):
    """
    Convenience function.
    """
    try:
        return span.ents[0].label_ == "PERSON"
    except IndexError:
        return False


def get_manual_speaker_cluster(quote, cluster):
    """
    If the match doesn't have a cluster, find any speakers in clusters that match manually.

    The idea here is if the speaker is "Rosenberg" to pull "Detective Jeff Rosenberg" from the clusters.

    TODO: Add this back into attribution code, and a test.
    """
    if any([len(quote.speaker) > 1, quote.speaker[0].pos_ != "PRON"]):
        speaker = " ".join([s.text for s in quote.speaker])
        for span in cluster:
            if speaker in span.text:
                return True
    else:
        return False


def compare_quote_to_cluster(
    quote: DQTriple,
    cluster: SpanGroup,
):
    """
    Finds first span in cluster that matches (according to compare_quote_to_cluster_member) with provided quote.

    TODO: Doesn't consider further matches. Is this a problem?

    Input:
        quote - textacy quote object
        cluster - coref cluster

    Output:
        cluster_index (int) - index of cluster member that matches quote (or -1, if none match)
    """
    try:
        return next(
            cluster_index
            for cluster_index, cluster_member in enumerate(cluster)
            if compare_quote_to_cluster_member(quote, cluster_member)
        )
    except StopIteration:
        return -1


def compare_quote_to_cluster_member(quote: DQTriple, span: Span):
    """
    Compares the starting character of the quote speaker and the cluster member as well as the quote speaker sentence and the cluster member sentence to determine equivalence.

    Input:
        q (quote triple) - one textacy quote triple
        cluster_member - one spacy-parsed entity cluster member

    Output:
        bool
    """
    # filters out very short strings
    if span[0].pos_ != "PRON" and len(span) < 2 and len(span[0]) < 4:
        return False
    if abs(quote.speaker[0].sent.start_char - span.sent.start_char) < MIN_SPEAKER_DIFF:
        if abs(quote.speaker[0].idx - span.start_char) < MIN_SPEAKER_DIFF:
            return True
    if span.start_char <= quote.speaker[0].idx:
        if span.end_char >= (quote.speaker[-1].idx + len(quote.speaker[-1])):
            return True
    return False


def compare_spans(
    s1: Span,
    s2: Span,
) -> bool:
    """
    Compares two spans to see if their starts and ends are less than min_entity_diff.

    If compare

    Input:
        s1 and s2 - spacy spans
        min_entity_diff (int) - threshold for difference in start and ends
    Output:
        bool - whether the two spans start and end close enough to each other to be "equivalent"

    """
    return all(
        [
            abs(getattr(s1, attr) - getattr(s2, attr)) < MIN_ENTITY_DIFF
            for attr in ["start", "end"]
        ]
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


# VIZ
def generate_code(n: int, label: str, start: bool = True, color_key: dict = {}) -> str:
    if label in color_key:
        color = color_key.get(label)
    else:
        color = "PapayaWhip"
    if label.lower() == "quote":
        if start:
            return f'<a name="quote{n}"></a><a href="#{n}"><span id="QUOTE" style="background-color: {color};">'
        else:
            return "</span></a>"
    else:
        if start:
            return f'<span id="{label}" style="color: {color};">'
        else:
            return "</span>"
