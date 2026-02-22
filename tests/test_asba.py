from src.lexicon_absa import LexiconABSA
from src.transformer_absa import ML_ABSA
from src.llm_asba import LLMABSA
analyzers = [LexiconABSA(), ML_ABSA(), LLMABSA(), LLMABSA("mistral:7b")]

#Sample tests here(for debugging purposes)
text = ["The pizza was delicious but the service is terrible.","Nice park with fun roller coasters and also fun with smaller children. Negatives..Employees were annoying."
        , "The food is uniformly exceptional, with a very capable kitchen which will proudly whip up whatever you feel like eating, whether it's on the menu or not."
        ,"It dies after just a few hours.", "The pizza was so good i wanted to throw it in the trash."]

for model in analyzers:
    print("Model: ", model.name)
    for f in text:
        print("Sentence: ", f, " \n \n", [t for t in model.analyze(f)])
