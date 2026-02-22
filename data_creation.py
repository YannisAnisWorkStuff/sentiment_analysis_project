# data_creation.py (CHATGPT GENERATED THIS CODE, ALL CREDIT GOES TO IT)
import pandas as pd
import random
import csv
import os

def sample_sentences(df, n=5):
    """Return n random unique sentences (and all associated aspect rows)."""
    unique_ids = df["id"].unique().tolist()
    sampled_ids = random.sample(unique_ids, min(n, len(unique_ids)))
    return df[df["id"].isin(sampled_ids)]

def create_custom_examples():
    """Return list of dicts for 9 handcrafted examples (same format)."""
    data = []

    # --- 3 simple cases ---
    simple_cases = [
        ("1",
         "The pizza was delicious.",
         "pizza", "positive", 4, 9),
        ("2",
         "The battery life is terrible.",
         "battery life", "negative", 4, 16),
        ("3",
         "The hotel room was clean.",
         "room", "positive", 10, 14),
    ]

    # --- 3 complex cases ---
    complex_cases = [
        ("4",
         "The phone screen is bright, but the battery drains quickly.",
         "screen", "positive", 10, 16),
        ("4",
         "The phone screen is bright, but the battery drains quickly.",
         "battery", "negative", 38, 45),

        ("5",
         "The laptop is light and fast but heats up under load.",
         "laptop", "positive", 4, 10),
        ("5",
         "The laptop is light and fast but heats up under load.",
         "laptop", "negative", 35, 41),

        ("6",
         "The wifi was slow, but at least it was free.",
         "wifi", "negative", 4, 8),
        ("6",
         "The wifi was slow, but at least it was free.",
         "wifi", "positive", 39, 43),
    ]

    # --- 3 edge cases (sarcasm / implicit / tricky) ---
    edge_cases = [
        ("7",
         "Oh great, another update that slows everything down.",
         "update", "negative", 11, 17),
        ("8",
         "The waiter was polite, if you consider ignoring us polite.",
         "waiter", "negative", 4, 10),
        ("9",
         "This phone could be worse, I guess that's something.",
         "phone", "neutral", 5, 10),
    ]

    for cid, sent, aspect, polarity, start, end in (simple_cases + complex_cases + edge_cases):
        data.append({
            "id": cid,
            "Sentence": sent,
            "Aspect Term": aspect,
            "polarity": polarity,
            "from": start,
            "to": end
        })
    return pd.DataFrame(data)


def main():
    # Check if dataset.csv already exists, if yes, do nothing
    if os.path.exists("../data/dataset.csv"):
        print(f"✅ Dataset already exists. Skipping creation.")
        return

    # --- Load datasets ---
    laptop_df = pd.read_csv("../data/Laptop_Train_v2.csv") #It's ../ because it will be executed by comparison.ipynb
    rest_df = pd.read_csv("../data/Restaurants_Train_v2.csv")

    # --- Sample 5 unique sentences from each (include all their aspect rows) ---
    laptop_sample = sample_sentences(laptop_df, n=5)
    rest_sample = sample_sentences(rest_df, n=5)

    # --- Combine the random and custom examples ---
    combined_df = pd.concat([laptop_sample, rest_sample], ignore_index=True)

    # Add custom 9 examples
    custom_df = create_custom_examples()
    combined_df = pd.concat([combined_df, custom_df], ignore_index=True)

    # --- Shuffle rows for randomness ---
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Save ---
    combined_df.to_csv("../data/dataset.csv", index=False, quoting=csv.QUOTE_ALL)
    print(f"✅ dataset.csv created with {len(combined_df)} rows.")


if __name__ == "__main__":
    main()
