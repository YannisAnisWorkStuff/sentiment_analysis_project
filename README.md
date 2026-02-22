# Aspect Based Sentiment Analysis Project
### Data 6 | Yiannis Ftiti

--- 
This project implements Aspect Based Sentiment Analysis using 3 different approaches:

**Lexicon-based**, **Transformer-based** & **LLM-based**

---

## Project Structure

```
Data6/
├── README.md (you are here)
├── requirements.txt (python requirements)
├── datacreation.py (python script used to create the test dataset)
├── data/
│   ├── dataset.csv (Sample dataset using the other 2 datasets)
│   ├── Laptop_Train_v2.csv
│   └── Restaurants_Train_v2.csv
├── notebooks/
│   ├── exploration.ipynb (Exploration & usage of all models in 3 different test cases)
│   └── comparison.ipynb (Test and comparason of all models using the test dataset)
├── src/
│   ├── __init__.py
│   ├── base.py (Base interface)
│   ├── lexicon_absa.py
│   ├── transformer_absa.py
│   ├── llm_absa.py
│   └── utils.py (Helper class)
├── tests/
│   └── test_asba.py (Python script used to test the models during development)
└── doc/
    ├── presentation slides.pptx (The presentation slides)
    └── report.pdf (In depth report)


```


## Installation

Download the zip file from canvas, and extract it.

(Optional) Create a virtual environment

```
python -m venv venv
source venv/bin/activate (for Windows: venv\Scripts\activate)
```

Install the required dependencies:

```
pip install -r requirements.txt
```
Next, download the spaCy English model

```
python -m spacy download en_core_web_sm
```

Finally, Install [Ollama](https://ollama.ai/) and the required local models:
```
ollama pull phi3
ollama pull mistral:7b
```
(this step is required in order to make the LLM based implementation function)


## Running the Implementations

To check if all implementations work, you can run the test:

```
python -m tests.test_asba
```

## Implementations

I implement 3 different ABSA methods, as per the project requires us.

- Rule based lexicon model
- Transformer based model
- LLM based model(using Phi-3 and Mistral 7B)

All implement the interface ABSAAnalyzer in base.py.

## Usage Example

```python
from src.lexicon_absa import LexiconABSA
from src.transformer_absa import ML_ABSA
from src.llm_asba import LLMABSA

#Initialize the models
lexicon_model = LexiconABSA()
ml_model = ML_ABSA()
llm_phi3 = LLMABSA()           # by default it will load the model phi-3
llm_mistral = LLMABSA("mistral:7b")  # larger model(you can use any model here)

#Example sentence
text = "The camera quality is amazing but the battery life is awful."

#Run analysis
print(lexicon_model.name)
for result in lexicon_model.analyze(text):
    print(result)

print(ml_model.name)
for result in ml_model.analyze(text):
    print(result)

print(llm_phi3.name)
for result in llm_phi3.analyze(text):
    print(result)

print(llm_mistral.name)
for result in llm_mistral.analyze(text):
    print(result)

```

Output:
```
LexiconABSA
AspectSentiment(aspect='quality', sentiment='positive', confidence=0.5859, text_span=(11, 18))
AspectSentiment(aspect='life', sentiment='negative', confidence=0.4588, text_span=(46, 50))

ML_ABSA
AspectSentiment(aspect='The camera quality', sentiment='positive', confidence=0.9980840682983398, text_span=(0, 18))
AspectSentiment(aspect='the battery life', sentiment='negative', confidence=0.9948299527168274, text_span=(34, 50))

LLMABSA (phi3)
AspectSentiment(aspect='camera quality', sentiment='positive', confidence=0.95, text_span=(4, 18))
AspectSentiment(aspect='battery life', sentiment='negative', confidence=0.97, text_span=(32, 42))

LLMABSA (mistral:7b)
AspectSentiment(aspect='camera quality', sentiment='positive', confidence=0.95, text_span=(4, 17))
AspectSentiment(aspect='battery life', sentiment='negative', confidence=0.9, text_span=(32, 42))
```

## Notebooks

I've created 2 notebooks:

- exploration.ipynb, Demonstrates & compares the 3 models on 3 sentences, each with a different case(Simple, Complex & Edge)


- comparison.ipynb – Evaluates & compares the 3 model's performance on the merged dataset.
## References & Credits

- [spaCy](https://spacy.io/): NLP preprocessing and dependency parsing

- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment): Lexicon based sentiment scoring

- [Hugging Face Transformers](https://huggingface.co/yangheng/deberta-v3-base-absa-v1.1): Pre trained ABSA model

- [Ollama](https://ollama.com/): Local LLM inference

- [SemEval 2014 ABSA Dataset](https://www.kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis): Used for evaluation and testing