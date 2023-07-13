"""
Cribbed from textacy!!!
"""
import pytest
import spacy
from sayswho.quotes import direct_quotations


@pytest.fixture(scope="module")
def nlp():
    return spacy.load("en_core_web_lg")


@pytest.mark.parametrize(
    "text, exp",
    [
        (
            'Burton said, "I love those cats!"',
            [(["Burton"], ["said"], '"I love those cats!"')],
        ),
        (
            'Burton explained from a podium. "I love those cats," he said.',
            [(["he"], ["said"], '"I love those cats,"')],
        ),
        (
            '"I love those cats!" insists Burton. "Yeah, I absolutely do."',
            [
                (["Burton"], ["insists"], '"I love those cats!"'),
                (["Burton"], ["insists"], '"Yeah, I absolutely do."'),
            ],
        ),
        (
            '"Some people say otherwise," he conceded.',
            [(["he"], ["conceded"], '"Some people say otherwise,"')],
        ),
        (
            'Burton claims that his favorite book is "One Hundred Years of Solitude".',
            [],
        ),
        (
            'Burton thinks that cats are "cuties".',
            [],
        ),
        (
            '"This is really a shame", said Officer Tusk of the Walrus Police Department. "He had such a life ahead of him"',
            [
                (["Officer", "Tusk"], ["said"], '"This is really a shame"'),
                (["Officer", "Tusk"], ["said"], '"He had such a life ahead of him"'),
            ],
        ),
    ],
)
def test_direct_quotations(nlp, text, exp):
    obs = list(direct_quotations(nlp(text)))
    assert all(
        hasattr(dq, attr) for dq in obs for attr in ["speaker", "cue", "content"]
    )
    obs_text = [
        ([tok.text for tok in speaker], [tok.text for tok in cue], content.text)
        for speaker, cue, content in obs
    ]
    assert obs_text == exp


@pytest.mark.parametrize(
    "text, speakers",
    [
        (  # tests that odd numbers of quotation marks can be parsed
            """He approached the stable. "Where are the horses' carrots?" he asked.""",
            ["he"],
        ),
        (  # tests that overlapping quotes are ignored
            "The stranger was eating a burger. He told everyone, \"This 'hamburger with extra cheese and pickles' is good.\"",
            ["He"],
        ),
        (  # tests ending quotes at linebreaks
            '\'uneasy\' on Gilmer street"\nPolice are investigating a shooting. They did not identify the 17-year-old because he is a minor.\n"Detectives are looking into it," Richmond police said in a news release Tuesday morning.',
            ["Richmond", "police"],
        ),
        (  # tests second use of windower
            "He pounces on counters and jabs the gun at tellers, only 2 pounds of pressure preventing that gun from firing.\n\"He's going to hurt somebody because he's carrying a revolver with the hammer cocked back. He's saying he's going to kill people,\" said Paul Martin, a Pinellas sheriff's detective who has been chasing the robber for two years. \"He's pointed guns at people's heads.\"\nThey call him the crowbar robber.",
            ["Paul", "Martin", "Paul", "Martin"],
        ),
        (  # checks that attributions don't leak over linebreaks (if they aren't supposed to)
            "And despite perhaps sometimes seeming like superheroes, sandwich aritsts are just like everyone else in society. Conney said in order to be what the people need, the job requires them to \"do their best to deal with it and fix whatever problem there is at that moment.\"\nThe chief said being involved in violence is never easy for anyone, but added that a sandwich artists' work isn't necessarily done when they leave the scene.",
            ["Conney"],
        ),
        (  # find quotes in last sentences
            'Garnier said he also learned a lesson from the ordeal.\n"Think before you act," the clown said. "Your actions have repercussions. No matter how trivial and joking I thought it was, people took it seriously."',
            ["clown", "clown"],
        ),
    ],
)
def test_adjustment_for_quote_detection(nlp, text, speakers):
    quotes = direct_quotations(nlp(text))
    assert [speaker.text for quote in quotes for speaker in quote.speaker] == speakers
