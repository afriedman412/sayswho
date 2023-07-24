import spacy
import numpy as np
import regex as re
from .quote_finder import quote_finder
from . import constants
from . import helpers
from jinja2 import Environment, FileSystemLoader


class SaysWho:
    """
    Main class for package. Instantiation loads spacy models so they don't have to be loaded again for repeat use.

    Input:
        text (str) - if provided, text will be analyzed on instantiation
        coref_nlp (str) - name of coref model
        base_nlp (str) - name of base model (for everything but coref) ... using en_core_web_lg because results are better than smaller models.
        prune (bool) - if True, outlying PERSONS will be removed from coref clusters via helpers.prune_cluster_people
        prep_text (bool) if True, text will be prepped for analysis via helpers.prep_text_for_quote_detection
    """

    def __init__(
        self,
        text: str = None,
        coref_nlp: str = "en_coreference_web_trf",
        base_nlp: str = "en_core_web_lg",
        prune: bool = True,
        prep_text: bool = True,
    ):
        for v in ["coref_nlp", "base_nlp"]:
            if not spacy.util.is_package(eval(v)):
                raise OSError(
                    f"SpaCy model {v} not installed. See README for instructions on how to install models."
                )
            self.__setattr__(v, spacy.load(eval(v)))
        self.prune = prune
        self.prep_text = prep_text
        if text:
            self.attribute(text)

    def expand_match(self, match=None):
        """
        Makes QuoteClusterMatch (or a list of QuoteClusterMatches) human-interpretable.

        Input:
            match (None, QuoteClusterMatch or list[QuoteClusterMatch]) - the QuoteClusterMatch(s) to be interpreted. Uses self.quote_matches if nothing is proivded.
        """
        if not match:
            match = self.quote_matches

        if isinstance(match, list):
            for m in match:
                self.expand_match(m)
        else:
            for m_ in ["quote", "cluster", "person"]:
                if getattr(match, f"{m_}_index", None) is not None:
                    i = eval(f"self.{m_}s")
                    v = getattr(match, f"{m_}_index")
                    data = helpers.format_cluster(i[v]) if m_ == "cluster" else i[v]
                    print(m_.upper(), f": {v}" "\n", data, "\n")
        return

    def attribute(self, text: str):
        """
        Top level function. Parses text, matches quotes to clusters and gets ent matches.
        Input:
            t (str) - text file to be analyzed and attributed

        Output:
            self.quote_matches (list[QuoteClusterMatch]) - list of quote/coref cluster match tuples
        """
        if self.prep_text:
            text = helpers.prep_text_for_quote_detection(text)
        self.parse_text(text)
        self.quote_matches = self.get_matches()
        return

    def parse_text(self, text: str):
        """
        Imports text, gets coref clusters, copies coref clusters, finds PERSONS and gets NER matches.

        Input:
            text (string) - text to be analyzed

        Ouput:
            self.coref_doc - spacy coref-parsed doc
            self.doc - spacy doc with coref clusters
            self.clusters - coref clusters
            self.quotes - list of textacy-extracted quotes
            self.persons - list of PERSON entities
        """
        # instantiate spacy doc
        self.coref_doc = self.coref_nlp(text)
        self.doc = self.base_nlp(text)

        # extract quotations
        self.quotes = [q for q in quote_finder(self.doc)]

        # extract coref clusters and clone to doc
        self.clusters = [
            helpers.clone_cluster(cluster, self.doc)
            for k, cluster in self.coref_doc.spans.items()
            if k.startswith("coref")
        ]
        if self.prune:
            self.clusters = [
                helpers.prune_cluster_people(cluster) for cluster in self.clusters
            ]

        self.persons = [e for e in self.doc.ents if e.label_ == "PERSON"]
        return

    def get_matches(self):
        """
        Master function to match quotes with coref clusters via matrix multiplication.

        Output:
            results (list) - list of QuoteClusterMatch tuples.
        """
        pairs_dicto = self.make_pairs()
        arrays = {k: self.make_matrix(k, v) for k, v in pairs_dicto.items()}

        big_matrix = np.concatenate(
            (
                np.transpose(
                    np.nonzero(
                        arrays["quotes_persons"].dot(arrays["clusters_persons"].T)
                    )
                ),
                np.transpose(np.nonzero(arrays["quotes_clusters"])),
            )
        )

        results = sorted(
            list(set([constants.QuoteClusterMatch(i, j) for i, j in big_matrix])),
            key=lambda m: m.quote_index,
        )

        return results

    def make_pairs(self) -> dict:
        """
        Creates quote/person, quote/cluster and cluster/person pairs for resolution and cleaning.

        TODO: Ensure pronouns aren't being skipped!
        TODO: Make ratio threshold a variable
        """
        if not all([v in self.__dict__ for v in ["quotes", "clusters", "persons"]]):
            raise Exception("No text parsed -- run SaysWho.attribute(text).")

        pairs_dicto = {
            p: []
            for p in [
                "quotes_persons",
                "quotes_clusters",
            ]
        }

        for quote_index, quote in enumerate(self.quotes):
            pairs_dicto["quotes_clusters"] += [
                (quote_index, cluster_index)
                for cluster_index, cluster in enumerate(self.clusters)
                for span in cluster
                if helpers.compare_quote_to_cluster_member(quote, span)
            ]

            pairs_dicto["quotes_persons"] += [
                (quote_index, person_index)
                for person_index, person in enumerate(self.persons)
                if helpers.span_contains(quote, person)
            ]

            pairs_dicto["quotes_clusters"] += [
                (quote_index, cluster_index)
                for cluster_index, cluster in enumerate(self.clusters)
                if quote_index
                not in [m[0] for m in set(pairs_dicto["quotes_clusters"])]
                if quote.speaker[0].text in [p.text for p in self.persons]
                if helpers.get_manual_speaker_cluster(quote, cluster)
            ]

        pairs_dicto["clusters_persons"] = [
            (cluster_index, person_index)
            for person_index, person in enumerate(self.persons)
            for cluster_index, cluster in enumerate(self.clusters)
            for span in cluster
            if helpers.span_contains(person, span)
            if not helpers.pronoun_check(span)
        ]
        return pairs_dicto

    def make_matrix(self, key: str, pairs: list[tuple]) -> np.array:
        """
        Convenience function for converting quote/cluster, quote/person and cluster/person pairs into binary matrices.

        Input:
            key (str) - data types in pairs, connected by "_" (ie "quotes_clusters" means the pairs data is (quote, cluster))
            pairs (list[tuple]) - (data type 1, data type 2) matches

        Output:
            m (np.array) - binary matrix of existing data type matches
        """
        x, y = key.split("_")
        m = np.zeros([len(self.__getattribute__(_)) for _ in [x, y]])
        for i, j in pairs:
            m[i, j] = 1
        return m

    def print_clusters(self):
        """
        Print clusters with duplicate text removed. For easier interpretation!
        """
        for n, cluster in enumerate(self.clusters):
            print(n, set(t.text for t in cluster))

    # VIZ CODE
    def process_quote_for_rendering(self, quote_match):
        quote = {
            "content": self.quotes[quote_match.quote_index].content,
            "cue": "".join(
                [t.text_with_ws for t in self.quotes[quote_match.quote_index].cue]
            ),
            "cluster": ", ".join(
                set(
                    [
                        c.text
                        for c in self.clusters[quote_match.cluster_index]
                        if c[0].pos_ != "PRON"
                    ]
                )
            ),
        }
        return quote

    def yield_quotes(self):
        for quote_index in range(len(self.quotes)):
            for m in (qm for qm in self.quote_matches if qm.quote_index == quote_index):
                base_dict = self.process_quote_for_rendering(m)
                base_dict["cluster_index"] = m.cluster_index
                base_dict["cluster"] = ", ".join(
                    set(
                        [
                            c.text
                            for c in self.clusters[m.cluster_index]
                            if c[0].pos_ != "PRON"
                        ]
                    )
                )
            yield (base_dict)

    def process_text_into_html(self):
        quote_indexes = [
            ((q.content.start, q.content.end), "QUOTE", n)
            for n, q in enumerate(self.quotes)
        ]
        html_output = "<p>"
        color_key = {}
        for token in self.doc:
            coded = False
            if token.text == "\n":
                token_text = "<br>"
            else:
                token_text = token.text_with_ws
            for index_match in (i_ for i_ in quote_indexes if token.i in i_[0]):
                coded = True
                match_indexes, label, n = index_match
                start = not match_indexes.index(token.i)
                html_output += helpers.generate_code(n, label, start, color_key)
                html_output += token_text
            if not coded:
                html_output += token_text

        html_output += "</p>"
        return html_output

    def render_to_html(
        self,
        article_title: str = "My Article",
        output_path: str = "temp.html",
        save_file: bool = True,
    ):
        metadata = {
            "title": article_title,
            "bodytext": self.process_text_into_html(),
            "quotes": list(self.yield_quotes()),
        }
        rendered = (
            Environment(loader=FileSystemLoader("./"))
            .get_template("template.html")
            .render(metadata)
        )

        if save_file:
            if not output_path.endswith(".html"):
                output_path = output_path + ".html"
            with open(output_path, "w+") as f:
                try:
                    f.write(rendered)
                except UnicodeDecodeError:
                    rendered = re.sub("\u2014", "-", rendered)
                    f.write(rendered)
            return

        else:
            return rendered
