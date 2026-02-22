from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class AspectSentiment:

    """Data class for aspect-sentiment pairs"""
    aspect: str               # The aspect/feature mentioned
    sentiment: str            # Sentiment label: 'positive', 'negative', 'neutral'
    confidence: float         # Confidence score (0.0 to 1.0)
    text_span: Tuple[int, int] = None  # Optional: (start, end) position in original text


class ABSAAnalyzer:

    """Base class/interface for all ABSA implementations"""

    def analyze(self, text: str) -> List[AspectSentiment]:
        """
        Analyze text and extract aspect-sentiment pairs

        Args:
        text: Input text to analyze

        Returns:
        List of AspectSentiment objects
        """
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self.__class__.__name__