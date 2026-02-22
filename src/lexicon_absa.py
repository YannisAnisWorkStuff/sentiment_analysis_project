import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List
from .base import ABSAAnalyzer, AspectSentiment
from .utils import label_from_score
from .utils import negation

"""
Lexicon based aspect based Sentiment Analysis
"""

class LexiconABSA(ABSAAnalyzer):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vader = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        results = []

        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                aspect = token.text
                opinion = None
                score = 0.0

                for child in token.children:
                    if child.dep_ == "amod" and child.pos_ == "ADJ":
                        opinion = child.text
                        score = self.vader.polarity_scores(opinion)["compound"]
                        score = negation(child, score)

                if opinion is None and token.dep_ == "nsubj" and token.head.pos_ == "AUX":
                    for child in token.head.children:
                        if child.dep_ == "acomp" and child.pos_ == "ADJ":
                            opinion = child.text
                            score = self.vader.polarity_scores(opinion)["compound"]
                            score = negation(child, score)

                if token.dep_ == "dobj" and token.head.pos_ == "VERB":
                    opinion_word = token.head.text
                    score = self.vader.polarity_scores(opinion_word)["compound"]
                    score = negation(token.head, score)

                if token.dep_ == "nsubjpass" and token.head.pos_ == "VERB":
                    advmod_opinion = [child for child in token.head.children if child.dep_ in {"advmod", "acomp"} and child.pos_ in {"ADJ", "ADV"}]
                    for op in advmod_opinion:
                        opinion = op.text
                        score = self.vader.polarity_scores(opinion)["compound"]
                        score = negation(op, score)


                if opinion:
                    sentiment = label_from_score(score)
                    confidence = abs(score)
                    results.append(
                        AspectSentiment(
                            aspect=aspect,
                            sentiment=sentiment,
                            confidence=confidence,
                            text_span=(token.idx, token.idx + len(aspect)),
                        )
                    )

        return results