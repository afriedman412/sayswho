import pytest
from sayswho import SaysWho
import os

@pytest.fixture(scope="module")
def says_who_ner_loaded(load_ner_nlp):
    # load_ner_nlp returns path to downloaded model
    ner_model_path = os.path.join(load_ner_nlp, "nba-model-best/")
    sw = SaysWho(ner_nlp=ner_model_path)
    return sw

@pytest.fixture(scope="module")
def ner_test_text():
    test_text = """"It was a rough night," said Carlos Boozer of the Orlando Magic, who only managed 9 points in a loss to the Celtics. He was distracted by his new haircut, which had been the subject of much ridicule on the internet."""
    return test_text

def test_ner(says_who_ner_loaded):
    # tests whether the bool property of SW is triggered
    assert says_who_ner_loaded.ner == True

def test_ner_model(says_who_ner_loaded, ner_test_text):
    # tests whether NER model works
    ner_doc = says_who_ner_loaded.ner_nlp(ner_test_text)
    ents = [(str(e), e.label_) for e in ner_doc.ents]
    assert ents == [
        ('Carlos Boozer', 'PLAYER'),
        ('the Orlando Magic', 'TEAM'),
        ('the Celtics.', 'TEAM')]

def test_ner_attribution(says_who_ner_loaded, ner_test_text):
    # tests quote cluster matching
    says_who_ner_loaded.attribute(ner_test_text)
    quote_cluster_match = says_who_ner_loaded.quote_matches[0]
    assert quote_cluster_match.quote_index == 0
    assert quote_cluster_match.cluster_index == 0
    assert set(quote_cluster_match.ents) == set(('PLAYER', 'TEAM'))