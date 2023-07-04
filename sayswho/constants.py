from spacy.symbols import (aux, auxpass, csubj, dobj, neg, nsubj)
import json
from collections import namedtuple
from spacy.tokens import Span

# TODO: adjust this for remote functionality
json_path = "../CJJ/query_work_files/query_results_2_2_23/"
file_key = json.load(open('./sayswho/doc_file_key.json'))
ner_nlp = "./output/model-last/"

QuoteClusterMatch: tuple[int, int] = namedtuple(
    "QuoteClusterMatch", 
    ["quote_index", "cluster_index"], 
    defaults=(None, None)
)

QuoteEntMatch: tuple[int, int, int, int] = namedtuple(
    "QuoteEntMatch", 
    ["quote_index", "cluster_index", "person_index", "ent_index"],
    defaults=(None, None, None, None)
)

ClusterEntMatch: tuple[int, int, Span, int] = namedtuple(
    "ClusterEntMatch",
    ["cluster_index", "span_index", "span", "ent_start"]
)

EvalResults: tuple[int, int, int] = namedtuple(
    "EvalResults", ["n_quotes", "n_ent_quotes", "n_ents_quoted"]
    )

Boundaries: tuple[int, int] = namedtuple(
    "boundaries",
    ['start', 'end']
)


color_key={
    "QUOTE": "lightyellow",
    "LAW ENFORCEMENT": "maroon"
}

"""
Constants for ent_like matching
"""
ent_like_words = [
    "police",
    "police sources",
    "police spokesperson",
    "law enforcement sources",
    "Detective",
    "Sergeant",
    "Judge",
    "prosecutors",
    "U.S. attorney",
    "warrant",
    "detective"
]

"""
Constants for token/entity matching
"""
MIN_SPEAKER_DIFF = 5
MIN_ENTITY_DIFF = 2
MIN_QUOTE_LENGTH = 3

"""
Constants for textacy quote identification
"""
_ACTIVE_SUBJ_DEPS = {csubj, nsubj}
_VERB_MODIFIER_DEPS = {aux, auxpass, neg}

"""
For prepping text for quote detection.
"""
ALL_QUOTES = '‹「`»」‘"„›”‚’\'』『«“'
DOUBLE_QUOTES = '‹「」»"„『”‚』›«“'
BRACK_REGEX = r"[{}]"
DOUBLE_QUOTES_NOSPACE_REGEX = r"(?<=\S)([{}])(?=\S)".format(DOUBLE_QUOTES)

"""
Ordinal points of the token.is_quote characters, matched up by start and end.

source:
switch = "\"\'"
start = "“‘```“‘«‹「『„‚"
end = "”’’’’”’»›」』”’"

"""
QUOTATION_MARK_PAIRS = {
    (34, 34), # " "
    (39, 39), # ' '
    (96, 8217),
    (171, 187),
    (8216, 8217),
    (8218, 8217),
    (8220, 8221),
    (8222, 8221),
    (8249, 8250),
    (12300, 12301),
    (12302, 12303),
    (8220, 34),
    (8216, 34),
    (96, 34),
    (171, 34),
    (8249, 34),
    (12300, 34),
    (12302, 34),
    (8222, 34),
    (8218, 34),
    (34, 8221),
    (34, 8217), # " ’
    (34, 10),
    (39, 10),
    (96, 10),
    (171, 10),
    (8216, 10),
    (8218, 10),
    (8249, 10)
    }

_reporting_verbs = {
        "according",
        "accuse",
        "acknowledge",
        "add",
        "admit",
        "agree",
        "allege",
        "announce",
        "argue",
        "ask",
        "assert",
        "believe",
        "blame",
        "charge",
        "cite",
        "claim",
        "complain",
        "concede",
        "conclude",
        "confirm",
        "contend",
        "continue",
        "criticize",
        "declare",
        "decline",
        "deny",
        "describe",
        "disagree",
        "disclose",
        "estimate",
        "explain",
        "fear",
        "hope",
        "insist",
        "maintain",
        "mention",
        "note",
        "observe",
        "order",
        "post",
        "predict",
        "promise",
        "read",
        "recall",
        "recommend",
        "reply",
        "report",
        "say",
        "scream",
        "state",
        "stress",
        "suggest",
        "tell",
        "testify",
        "think",
        "tweet",
        "urge",
        "warn",
        "worry",
        "write",
    }

