def label_from_score(score: float) -> str:
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def negation(token, score: float) -> float:
    for child in token.children:
        if child.dep_ == "neg":
            return score * -1
    return score

import re, json

def load_clean_json(content: str):
    content = content.strip().strip("`").strip()
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        content = match.group(0)
    content = re.sub(r",(\s*[\]}])", r"\1", content)
    return json.loads(content)
