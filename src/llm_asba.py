import json
import time
from typing import List
from ollama import Client
from .base import ABSAAnalyzer, AspectSentiment
from .utils import load_clean_json

"""
Aspect-Based Sentiment Analysis using a local LLM via Ollama.
"""

class LLMABSA(ABSAAnalyzer):

    def __init__(self, model: str = "phi3", tries: int = 3):
        self.client = Client()
        self.tries = tries
        self.model = model

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__} ({self.model})"

    def generate_prompt(self, text: str) -> str:
        prompt = """
        You are an aspect based sentiment analysis tool.
        Extract all aspects (topics, entities, or features) from the sentence and assign each one a sentiment label: 
        "positive", "negative", or "neutral".

        For each aspect, return:
        - "aspect": string  
        - "sentiment": string ("positive", "negative", or "neutral")  
        - "confidence": float between 0 and 1  
        - "text_span": [start, end] indices in the input string  

        Detect sarcasm and irony: 
        If a phrase sounds positive but actually conveys a mocking or negative tone, classify it as "negative".

        Respond with ONLY a valid JSON array, No explanations, no extra text.
        
        -------------------------
        Examples:
        
        Input: "The hotel room was spotless and the staff were friendly."
        Output:
        [
            {"aspect": "hotel room", "sentiment": "positive", "confidence": 0.9, "text_span": [4, 14]},
            {"aspect": "staff", "sentiment": "positive", "confidence": 0.88, "text_span": [33, 38]}
        ]

        Input: "The phone screen is bright but the battery drains quickly."
        Output:
        [
            {"aspect": "phone screen", "sentiment": "positive", "confidence": 0.85, "text_span": [4, 16]},
            {"aspect": "battery", "sentiment": "negative", "confidence": 0.9, "text_span": [33, 40]}
        ]

        Input: "This blender is perfect if you want to make noise louder than your neighbor’s lawnmower."
        Output:
        [
            {"aspect": "blender noise", "sentiment": "negative", "confidence": 0.93, "text_span": [5, 12]},
            {"aspect": "neighbor’s lawnmower", "sentiment": "negative", "confidence": 0.85, "text_span": [45, 64]}
        ]

        -------------------------
        Analyze the following sentence and reply ONLY with valid JSON:
        """
        return f"{prompt}\nInput: \"{text}\"\nOutput:"

    def analyze(self, text: str) -> List[AspectSentiment]:
        prompt = self.generate_prompt(text)

        for attempt in range(self.tries):
            try:
                response = self.client.generate(model=self.model, prompt=prompt)
                content = response["response"].strip()
                data = load_clean_json(content)
                results = []
                for item in data:
                    aspect = item.get("aspect", "")
                    sentiment = item.get("sentiment", "neutral").lower()
                    confidence = float(item.get("confidence", 0.8))
                    span = item.get("text_span", None)

                    if isinstance(span, list) and len(span) == 2:
                        text_span = tuple(span)
                    else:
                        text_span = None

                    results.append(
                        AspectSentiment(
                            aspect=aspect,
                            sentiment=sentiment,
                            confidence=confidence,
                            text_span=text_span,
                        )
                    )

                return results
            except Exception as e:
                print(f"\n Exception {e.__class__.__name__} thrown.. ({e}). Retrying... \n")
                time.sleep(1)
        return []