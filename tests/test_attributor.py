import pytest
import spacy
from sayswho.sayswho import Attributor

spacy.prefer_gpu()


@pytest.fixture(scope="module")
def a():
    return Attributor()

def test_attribution(a):
    test_text = open("./tests/qa_test_file.txt").read()
    a.attribute(test_text)

    assert len(a.quotes) == 4
    assert [
        " ".join([x.text for x in q.speaker])
        for q in a.quotes
    ] == ['he', 'he', 'she', 'Rogers']

    assert [
        " ".join([x.text for x in q.cue])
        for q in a.quotes
    ] == ['said', 'said', 'noted', 'added']

    assert [
        q.content.text for q in a.quotes
        ] == [
            '“Based on the actions the group was making, based on everything the gentlemen who came in had told me — if I allowed anyone in the store, they would try to cause harm to people,”',
            '“I couldn’t see how big the group was. I thought, ’It’s just seven to 10 people. Maybe they’ll back off.’”',
            '“They were all late teens, early 20s, clean-cut, typical blondie, blue-eyed, wholesome Utah boys,”',
            '“Then this African-American guy came from the restaurant,”'
    ]