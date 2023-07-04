import pytest
import spacy
from sayswho.sayswho import Attributor, evaluate
from sayswho.constants import EvalResults

spacy.prefer_gpu()

@pytest.fixture(scope="module")
def a_ner():
    return Attributor()

@pytest.mark.parametrize(
    "file_name, eval_result",
    [
        ('60DY-RN71-DYTM-N0M2-00000-00', EvalResults(n_quotes=0, n_ent_quotes=0, n_ents_quoted=0)),
        ('5V2D-0531-JBM5-R4GP-00000-00', EvalResults(n_quotes=0, n_ent_quotes=0, n_ents_quoted=0)),
        ('620B-S7K1-DY37-F2V0-00000-00', EvalResults(n_quotes=3, n_ent_quotes=3, n_ents_quoted=1)),

        # remove duplicate ents
        ('5V5C-JVP1-DY37-F4FC-00000-00', EvalResults(n_quotes=15, n_ent_quotes=5, n_ents_quoted=1)),
        ('65SG-FK31-JC3R-B541-00000-00', EvalResults(n_quotes=8, n_ent_quotes=6, n_ents_quoted=1)),

        # multiple ents per quote (??)
        ('5X5X-VPT1-DYNS-33VS-00000-00', EvalResults(n_quotes=1, n_ent_quotes=1, n_ents_quoted=1)),

        # to test span_contains
        ('5W5S-BDS1-JBM5-R229-00000-00', EvalResults(n_quotes=2, n_ent_quotes=2, n_ents_quoted=1)),

        # bad ordinals
        ('5W64-5441-DYTG-S3T7-00000-00', EvalResults(n_quotes=21, n_ent_quotes=2, n_ents_quoted=1)),

        # quote detection, min_length
        ('65RN-WFC1-DY8S-B509-00000-00', EvalResults(n_quotes=6, n_ent_quotes=0, n_ents_quoted=0)),
        ('5V49-J6V1-JC0C-J46G-00000-00', EvalResults(n_quotes=3, n_ent_quotes=0, n_ents_quoted=0)),
        ('5WXW-GBY1-JBRS-Y2HX-00000-00', EvalResults(n_quotes=7, n_ent_quotes=1, n_ents_quoted=1)),

        # paragraph issue fixed
        ('5X4C-PYG1-DYJT-21WJ-00000-00', EvalResults(n_quotes=6, n_ent_quotes=2, n_ents_quoted=2)),

        # "good not great" (??)
        ('5SVW-WWP1-JC6P-C282-00000-00', EvalResults(n_quotes=15, n_ent_quotes=2, n_ents_quoted=1)),

        # negative controls (lots of "police" but no ents)
        ('601N-VP41-JC8R-301M-00000-00', EvalResults(n_quotes=18, n_ent_quotes=0, n_ents_quoted=0)),

        # # missed a quote and a big ent first time around
        # ('642J-V391-DYRK-B1WM-00000-00', EvalResults(n_quotes=49, n_ent_quotes=6, n_ents_quoted=4)),

        # big boundary issues -- use for quote detection testing
        ('5TJH-PJG1-DY6J-M3TM-00000-00', EvalResults(n_quotes=15, n_ent_quotes=5, n_ents_quoted=2))
    ]
)

def test_ents(file_name, eval_result, a_ner):
    a_ner.attribute(open(f"./tests/quote_parse_test_files/{file_name}.txt").read())
    assert a_ner.ner
    e = evaluate(a_ner)
    print(a_ner.quotes)
    print(a_ner.ent_matches)
    assert evaluate(a_ner) == eval_result