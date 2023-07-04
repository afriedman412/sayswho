import pytest
from sayswho.sayswho import Attributor

@pytest.fixture(scope="module")
def a_ner():
    return Attributor(ner_nlp="./output/model-last/")

@pytest.mark.parametrize(
    "text, ents",
    [
        (
            '"This is really a shame", said Officer Tusk of the Walrus Police Department. "He had such a life ahead of him"',
            ["Officer Tusk of the Walrus Police Department."]
        ),
        (
            """Following the arrest, Loden was held in the Union County Jail for security reasons. After meeting with his wife on June 30, 2000, Loden waived his Miranda rights and confessed to a pair of Mississippi Bureau of Investigation officers. He said he killed the youth to preserve his public appearance.

"Looking back now, I wouldn't have released her because I would've lost the image of being the picture-perfect Marine," Loden said to investigators. "When I woke up, I saw the body. I knew I had done it."

An Itawamba County grand jury indicted Loden five months later. Despite the confession, the video tape and a mountain of physical evidence, he pleaded not guilty at his Nov. 21, 2000, arraignment.""",
        ["Mississippi Bureau of Investigation"]
        ),
        ("""The NYPD says it has in place careful guidelines for using facial recognition, and a "hit" off a database search is just a lead and does not automatically trigger that person's arrest.

"No one has ever been arrested based solely on a positive facial recognition match," said Assistant Commissioner Devora Kaye, an NYPD spokeswoman. "It is merely a lead, not probable cause. We are comfortable with this technology because it has proven to be a valuable investigative method."

Police also said a kid mug shot is kept only if the suspect is classified a juvenile delinquent and the case ends with a felony conviction.""",
        ["NYPD", "Assistant Commissioner Devora Kaye, an NYPD spokeswoman"]
        )
    ]
)

def test_ents(text, ents, a_ner):
    a_ner.attribute(text)
    assert a_ner.ner
    assert [e.text for e in a_ner.ents] == ents