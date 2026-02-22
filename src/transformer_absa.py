from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
from .base import ABSAAnalyzer, AspectSentiment
import spacy

"""
Transformer Aspect Based Sentiment Analysis using this pretrained model: yangheng/deberta-v3-base-absa-v1.1
"""

class ML_ABSA(ABSAAnalyzer):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.model = AutoModelForSequenceClassification.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
        self.model.eval()
        self.nlp = spacy.load("en_core_web_sm")

    def analyze(self, text: str) -> List[AspectSentiment]:
        doc = self.nlp(text)
        results = []

        aspects = [chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ in {'NOUN', 'PROPN'} and not chunk.root.is_stop]
        aspects = list(dict.fromkeys(aspects))

        for aspect in aspects:

            encoded = self.tokenizer(text, aspect, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**encoded)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]

            label_id = torch.argmax(probs).item()
            confidence = probs[label_id].item()

            sentiment = self.model.config.id2label[label_id]
            aspect_span = (text.lower().find(aspect.lower()), text.lower().find(aspect.lower()) + len(aspect))

            results.append(
                AspectSentiment(
                    aspect=aspect,
                    sentiment=str(sentiment).lower(),
                    confidence=confidence,
                    text_span=aspect_span,
                )
            )

        return results