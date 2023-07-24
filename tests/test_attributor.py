import pytest
import spacy
from sayswho import SaysWho

spacy.prefer_gpu()


@pytest.fixture(scope="module")
def says_who_loaded():
    test_text = open("./tests/qa_test_file.txt").read()
    sw = SaysWho(test_text)
    return sw


def test_find_all_quotes(says_who_loaded):
    assert len(says_who_loaded.quotes) == 4


def test_correct_speakers(says_who_loaded):
    assert [" ".join([x.text for x in q.speaker]) for q in says_who_loaded.quotes] == [
        "he",
        "he",
        "she",
        "Rogers",
    ]


def test_correct_actions(says_who_loaded):
    assert [" ".join([x.text for x in q.cue]) for q in says_who_loaded.quotes] == [
        "said",
        "said",
        "noted",
        "added",
    ]


def test_correct_content(says_who_loaded):
    assert [q.content.text for q in says_who_loaded.quotes] == [
        "“Based on the actions the group was making, based on everything the gentlemen who came in had told me — if I allowed anyone in the store, they would try to cause harm to people,”",
        "“I couldn’t see how big the group was. I thought, ’It’s just seven to 10 people. Maybe they’ll back off.’”",
        "“They were all late teens, early 20s, clean-cut, typical blondie, blue-eyed, wholesome boys,”",
        "“Then this African-American guy came from the restaurant,”",
    ]


def test_html_viz(says_who_loaded):
    expected_html = open("./tests/test_viz.html").read()
    test_html = says_who_loaded.render_to_html(save_file=False)
    assert len(test_html) == len(expected_html)
