# SaysWho
SaysWho is a Python package for identifying and attributing quotes in text. It uses [a combination of grammar and logic](https://textacy.readthedocs.io/en/latest/) to find quotes and their speakers, then uses a [coreferencing model](https://explosion.ai/blog/coref) to better clarify who is speaking. 

## Installation
Install and update using [pip](https://pip.pypa.io/en/stable/):
```
$ pip install sayswho
```

You will probably need to install the coreferencing model manually, then re-update SpaCy.

```
$ pip install pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl
$ pip install spacy -U
```

You also might need to download the main large SpaCy english model.
```
$ spacy download en_core_web_lg
```

## Notes
- Coreferencing (which this package uses to resolve speakers) is an experimental feature not fully implemented in SpaCy. SaysWho uses a pretrained model which SpaCy made available, but it was built with older code. This explains the version roulette required on installation.

## A Simple Example


```python
from sayswho.sayswho import Attributor
from sayswho import quote_helpers
test_text = open("./tests/qa_test_file.txt").read()
test_text = quote_helpers.prep_text_for_quote_detection(test_text)
```

#### Instantiate `Attributor` and run `.attribute` on target text.
This will probably raise an warning about differing SpaCy models, which you can ignore. 


```python
a = Attributor()
a.attribute(test_text)
```

    /usr/local/lib/python3.11/site-packages/spacy/util.py:910: UserWarning: [W095] Model 'en_coreference_web_trf' (3.4.0a2) was trained with spaCy v3.3 and may not be 100% compatible with the current version (3.6.0). If you see errors or degraded performance, download a newer compatible model or retrain your custom model with the current spaCy version. For more details and available updates, run: python -m spacy validate
      warnings.warn(warn_msg)
    /usr/local/lib/python3.11/site-packages/spacy/util.py:910: UserWarning: [W095] Model 'en_core_web_lg' (3.5.0) was trained with spaCy v3.5 and may not be 100% compatible with the current version (3.6.0). If you see errors or degraded performance, download a newer compatible model or retrain your custom model with the current spaCy version. For more details and available updates, run: python -m spacy validate
      warnings.warn(warn_msg)


#### See speaker, cue and content of every quote with `.quotes`.


```python
a.quotes
```




    [DQTriple(speaker=[he], cue=[said], content=“Based on the actions the group was making, based on everything the gentlemen who came in had told me — if I allowed anyone in the store, they would try to cause harm to people,”),
     DQTriple(speaker=[he], cue=[said], content=“I couldn’t see how big the group was. I thought, ’It’s just seven to 10 people. Maybe they’ll back off.’”),
     DQTriple(speaker=[she], cue=[noted], content=“They were all late teens, early 20s, clean-cut, typical blondie, blue-eyed, wholesome Utah boys,”),
     DQTriple(speaker=[Rogers], cue=[added], content=“Then this African-American guy came from the restaurant,”)]



#### See resolved entity clusters with `.clusters`.


```python
a.clusters
```




    [[four frightened, breathless men,
      They,
      they,
      them,
      the group,
      they,
      the group,
      they,
      the group,
      the group],
     [Terrance Mannery,
      Mannery,
      me,
      I,
      he,
      I,
      I,
      Mannery,
      Mannery,
      his,
      him,
      Mannery],
     [the Doki Doki dessert shop,
      Doki Doki,
      the store,
      the shop,
      Doki Doki,
      Doki Doki,
      the shop],
     [just seven to 10 people, that],
     [the Utah Pride Festival, Pride, Pride],
     [the mob, the crowd, They],
     [Michelle Turpin, who was walking west near Doki Doki with friends after Pride,
      she,
      Turpin],
     [Jen Parsons-Soren, who had attended Pride with Turpin,, she],
     [Lyft driver Ross Rogers, he, Rogers, Rogers],
     [the entrance of Doki Doki, the entrance],
     [this African-American guy, That],
     [a small group, the attackers],
     [the door, the door],
     [one of the attackers, the man]]



#### Use `.print_clusters()` to see unique text in each cluster, easier to read.


```python
a.print_clusters()
```

    0 {'them', 'the group', 'four frightened, breathless men', 'They', 'they'}
    1 {'I', 'his', 'Terrance Mannery', 'he', 'him', 'me', 'Mannery'}
    2 {'Doki Doki', 'the Doki Doki dessert shop', 'the shop', 'the store'}
    3 {'that', 'just seven to 10 people'}
    4 {'Pride', 'the Utah Pride Festival'}
    5 {'the crowd', 'They', 'the mob'}
    6 {'Michelle Turpin, who was walking west near Doki Doki with friends after Pride', 'she', 'Turpin'}
    7 {'she', 'Jen Parsons-Soren, who had attended Pride with Turpin,'}
    8 {'Rogers', 'Lyft driver Ross Rogers', 'he'}
    9 {'the entrance', 'the entrance of Doki Doki'}
    10 {'That', 'this African-American guy'}
    11 {'a small group', 'the attackers'}
    12 {'the door'}
    13 {'one of the attackers', 'the man'}


#### Quote/cluster matches are saved to `.quote_matches` as `namedtuples`.


```python
for qm in a.quote_matches:
    print(qm)
```

    QuoteClusterMatch(quote_index=0, cluster_index=1)
    QuoteClusterMatch(quote_index=1, cluster_index=1)
    QuoteClusterMatch(quote_index=2, cluster_index=6)
    QuoteClusterMatch(quote_index=3, cluster_index=8)


#### Use `.expand_match()` to interpret quote/cluster matches.


```python
a.expand_match()
```

    QUOTE : 0
     DQTriple(speaker=[he], cue=[said], content=“Based on the actions the group was making, based on everything the gentlemen who came in had told me — if I allowed anyone in the store, they would try to cause harm to people,”) 
    
    CLUSTER : 1
     ['Terrance Mannery', 'Mannery'] 
    
    QUOTE : 1
     DQTriple(speaker=[he], cue=[said], content=“I couldn’t see how big the group was. I thought, ’It’s just seven to 10 people. Maybe they’ll back off.’”) 
    
    CLUSTER : 1
     ['Terrance Mannery', 'Mannery'] 
    
    QUOTE : 2
     DQTriple(speaker=[she], cue=[noted], content=“They were all late teens, early 20s, clean-cut, typical blondie, blue-eyed, wholesome Utah boys,”) 
    
    CLUSTER : 6
     ['Michelle Turpin, who was walking west near Doki Doki with friends after Pride', 'Turpin'] 
    
    QUOTE : 3
     DQTriple(speaker=[Rogers], cue=[added], content=“Then this African-American guy came from the restaurant,”) 
    
    CLUSTER : 8
     ['Rogers', 'Lyft driver Ross Rogers'] 
    
