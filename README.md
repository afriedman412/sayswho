# SaysWho
**SaysWho** is a Python package for identifying and attributing quotes in text. It uses a combination of logic and grammer to find quotes and their speakers, then uses a [coreferencing model](https://explosion.ai/blog/coref) to better clarify who is speaking. It's built on [Textacy](https://textacy.readthedocs.io/en/latest/) and [SpaCy](https://spacy.io/).



## Notes
- Corefencing is an experimental feature not fully integrated into SpaCy, and the current pipeline is built on SpaCy 3.4. I haven't had any problems using it with SpaCy 3.5+, but it takes some finesse to navigate the different versions.


- I wrote this package as part of a larger project with a better-defined goal. The output of this version is kind of open-ended, and possible not as useful as it could be. HTML viz is coming, but I'm open to any suggestions about how this could be more useful!

## Installation
Install and update using [pip](https://pip.pypa.io/en/stable/):

```
$ pip install sayswho
```

You will probably need to install the coreferencing model manually, then re-update SpaCy. (see [Notes](#notes))

```
$ pip install pip install https://github.com/explosion/spacy-experimental/releases/download/v0.6.0/en_coreference_web_trf-3.4.0a0-py3-none-any.whl
$ pip install spacy -U
```

You also might need to download the main large SpaCy english model.

```
$ spacy download en_core_web_lg
```

## A Simple Example

##### Sample text adapted from [here](https://sports.yahoo.com/nets-jacque-vaughn-looking-forward-150705556.html):
> Nets Coach Jacque Vaughn was optimistic when discussing Ben Simmons's prospects on NBA TV.
> 
> “It’s been great, being able to check in with Ben," Vaughn said, via Nets Daily. “I look forward to coaching a healthy Ben Simmons. The team is excited to have him healthy, being part of our program and moving forward.
> 
> "He has an innate ability to impact the basketball game on both ends of the floor. So, we missed that in the Philly series and looking forward to it.”
> 
> Simmons arrived in Brooklyn during the 2021-22 season, but did not play that year after a back injury. The 26-year-old would make 42 appearances (33 starts) during a tumult-filled season for Brooklyn.
> 
> “He is on the court. No setbacks," Vaughn later told reporters about Simmons' workouts. “We’ll continue to see him improve through the offseason.”


#### Prep text with `prep_text_for_quote_detection()`, instantiate `Attributor` and run `.attribute` on target text.
`prep_text_for_quote_detection()` tweaks the text to avoid some common errors and generally improve results.

```python
from sayswho.sayswho import Attributor
from sayswho import quote_helpers

prepped_text = quote_helpers.prep_text_for_quote_detection(text)

a = Attributor()
a.attribute(prepped_text)
```


#### See speaker, cue and content of every quote with `.quotes`.


```python
print(a.quotes)
```

```
[DQTriple(speaker=[Vaughn], cue=[said], content=“It’s been great, being able to check in with Ben,"),
 DQTriple(speaker=[Vaughn], cue=[said], content=“I look forward to coaching a healthy Ben Simmons. The team is excited to have him healthy, being part of our program and moving forward."),
 DQTriple(speaker=[Vaughn], cue=[told], content=“He is on the court. No setbacks,"),
 DQTriple(speaker=[Vaughn], cue=[told], content=“We’ll continue to see him improve through the offseason.”)]
```



#### See resolved entity clusters with `.clusters`.


```python
print(a.clusters)
```

```
[[Ben Simmons's,
  Ben,
  a healthy Ben Simmons,
  him,
  He,
  Simmons,
  The 26-year-old,
  He,
  Simmons'x,
  him],
 [Nets Coach Jacque Vaughn, Vaughn, I, Vaughn],
 [Nets, The team, our, we],
 [an innate ability to impact the basketball game on both ends of the floor,
  that,
  it],
 [the 2021-22 season, that year],
 [Brooklyn, Brooklyn, We]]
```



#### Use `.print_clusters()` to see unique text in each cluster, easier to read.


```python
a.print_clusters()
```
```
0 {'Ben', 'He', 'The 26-year-old', 'a healthy Ben Simmons', "Simmons'x", "Ben Simmons's", 'Simmons', 'him'}
1 {'I', 'Nets Coach Jacque Vaughn', 'Vaughn'}
2 {'The team', 'our', 'we', 'Nets'}
3 {'it', 'an innate ability to impact the basketball game on both ends of the floor', 'that'}
4 {'that year', 'the 2021-22 season'}
5 {'Brooklyn', 'We'}
```


#### Quote/cluster matches are saved to `.quote_matches` as `namedtuples`.


```python
for qm in a.quote_matches:
    print(qm)
```
```
QuoteClusterMatch(quote_index=0, cluster_index=1)
QuoteClusterMatch(quote_index=1, cluster_index=1)
QuoteClusterMatch(quote_index=2, cluster_index=1)
QuoteClusterMatch(quote_index=3, cluster_index=1)
```


#### Use `.expand_match()` to view and interpret quote/cluster matches.


```python
a.expand_match()
```
```
QUOTE : 0
 DQTriple(speaker=[Vaughn], cue=[said], content=“It’s been great, being able to check in with Ben,") 

CLUSTER : 1
 ['Nets Coach Jacque Vaughn', 'Vaughn'] 

QUOTE : 1
 DQTriple(speaker=[Vaughn], cue=[said], content=“I look forward to coaching a healthy Ben Simmons. The team is excited to have him healthy, being part of our program and moving forward.") 

CLUSTER : 1
 ['Nets Coach Jacque Vaughn', 'Vaughn'] 

QUOTE : 2
 DQTriple(speaker=[Vaughn], cue=[told], content=“He is on the court. No setbacks,") 

CLUSTER : 1
 ['Nets Coach Jacque Vaughn', 'Vaughn'] 

QUOTE : 3
 DQTriple(speaker=[Vaughn], cue=[told], content=“We’ll continue to see him improve through the offseason.”) 

CLUSTER : 1
 ['Nets Coach Jacque Vaughn', 'Vaughn'] 
```

    

