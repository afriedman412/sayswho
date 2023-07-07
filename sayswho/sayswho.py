"""
Rewritten with less overhead.
"""
import spacy
from typing import Union, Iterable
import numpy as np
from .quotes import direct_quotations
from . import constants
from . import attribution_helpers

class Attributor:
    """
    TODO: Add manual speaker matching
    TODO: change self.clusters to be a list instead of a numerically indexed dict. (This is a holdover from the structure of the coref model output.)
    """
    def __init__(
            self, 
            coref_nlp: str="en_coreference_web_trf",
            base_nlp: str="en_core_web_lg",
            prune: bool=True,
            exp: bool=False
            ):
        self.coref_nlp = spacy.load(coref_nlp)
        self.base_nlp = spacy.load(base_nlp)
        self.prune = prune
        self.exp = exp
        
    def expand_match(match):
        def expando(match):
            for m_ in ['quote', 'cluster', 'person']:
                if getattr(match, f"{m_}_index", None) is not None:
                    i = eval(f"{m_}s")
                    v = getattr(match, f"{m_}_index")
                    data = attribution_helpers.format_cluster(i[v]) if m_=="cluster" else i[v]
                    print(m_.upper(), f": {v}""\n", data, "\n")

        if isinstance(match, list):
            for m in match:
                expando(m)
        else:
            expando(match)        
        
    def attribute(self, t: str):
        """
        Top level function. Parses text, matches quotes to clusters and gets ent matches.
        Input:
            t (str) - text file to be analyzed and attributed

        Output:
            self.quote_matches (list[QuoteClusterMatch]) - list of quotes 
        """
        self.parse_text(t)
        self.quote_matches = self.get_matches()

    def parse_text(self, t: str):
        """ 
        Imports text, gets coref clusters, copies coref clusters, finds PERSONS and gets NER matches.

        Input: 
            t (string) - formatted text of an article
            
        Ouput:
            self.coref_doc - spacy coref-parsed doc
            self.doc - spacy doc with coref clusters
            self.clusters - coref clusters
            self.quotes - list of textacy-extracted quotes
            self.persons - list of PERSON entities
        """
        # instantiate spacy doc
        self.coref_doc = self.coref_nlp(t)
        self.doc = self.base_nlp(t)

        # extract quotations
        self.quotes = [q for q in direct_quotations(self.doc, self.exp)]

        # extract coref clusters and clone to doc
        self.clusters = {
            int(k.split("_")[-1])-1: attribution_helpers.clone_cluster(cluster, self.doc) 
            for k, cluster in self.coref_doc.spans.items() 
            if k.startswith("coref")
            }
        if self.prune:
            self.clusters = {n:attribution_helpers.prune_cluster_people(cluster) for n, cluster in self.clusters.items()}
        
        self.persons = [e for e in self.doc.ents if e.label_=="PERSON"]
        return
    
    def get_matches(self):
        pairs_dicto = self.make_pairs()
        arrays = {k: self.make_matrix(k, v) for k,v in pairs_dicto.items()}
        
        return  sorted(
            list(set([
            constants.QuoteClusterMatch(i, j) for i,j in np.concatenate(
                (np.transpose(np.nonzero(arrays['quotes_persons'].dot(arrays['clusters_persons'].T))),
                 np.transpose(np.nonzero(arrays['quotes_clusters'])))
            )])), key=lambda m: m.quote_index
        )

    def make_pairs(self) -> dict:
        """
        Messy but lite ent finder. Easier than keeping track of all the ways to match ent and cluster.

        TODO: Ensure pronouns aren't being skipped!
        TODO: Make ratio threshold a variable
        """
        pairs_dicto = {p:[] for p in [
            'quotes_persons', 'quotes_clusters',
            ]}

        for quote_index, quote in enumerate(self.quotes):
            pairs_dicto['quotes_clusters'] += [
                (quote_index, cluster_index) 
                for cluster_index, cluster in self.clusters.items()
                for span in cluster
                if attribution_helpers.compare_quote_to_cluster_member(quote, span)
            ]
            
            pairs_dicto['quotes_persons'] += [
                (quote_index, person_index)
                for person_index, person in enumerate(self.persons)
                if attribution_helpers.span_contains(quote, person)
                ]
            
            pairs_dicto['quotes_clusters'] += [
                (quote_index, cluster_index) 
                for cluster_index, cluster in self.clusters.items()
                if quote_index not in [m[0] for m in set(pairs_dicto['quotes_clusters'])]
                if quote.speaker[0].text in [p.text for p in self.persons]
                if attribution_helpers.get_manual_speaker_cluster(quote, cluster)
                ]

        pairs_dicto['clusters_persons'] = [
            (cluster_index, person_index)
            for person_index, person in enumerate(self.persons)
            for cluster_index, cluster in self.clusters.items()
            for span in cluster
            if attribution_helpers.span_contains(person, span) 
            if not attribution_helpers.pronoun_check(span)
]
        return pairs_dicto
    
    def make_matrix(self, key, pairs):
        x, y = key.split("_")
        m = np.zeros([len(self.__getattribute__(_)) for _ in [x,y]])
        for i,j in pairs:
            m[i,j] = 1
        return m
    

