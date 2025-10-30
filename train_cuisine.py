# train_cuisine.py
"""
Train a cuisine classifier from a CSV and persist the model to cuisine_model.pkl.

Expected CSV columns:
 - 'ner_str' : string of normalized ingredients for each recipe (if missing, it will try to build from 'NER' or 'ingredients')
 - 'cuisine' : target label (optional). If missing or all NaN, the script will exit with a message.

Usage:
    python train_cuisine.py --csv RAW_recipes.csv --out cuisine_model.pkl --top-k 30
"""

import argparse
import os
import pickle
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def prepare_dataframe(path):
    df = pd.read_csv(path)
    # try to ensure ner_str exists
    if 'ner_str' not in df.columns:
        if 'NER' in df.columns:
            import ast, re
            def try_parse(x):
                if pd.isna(x):
                    return []
                try:
                    v = ast.literal_eval(x)
                    if isinstance(v, list):
                        return [str(i).lower().strip() for i in v if i]
                except Exception:
                    return [i.strip().lower() for i in str(x).split(',') if i.strip()]
                return []
            df['ner_str'] = df['NER'].apply(lambda lst: " ".join(try_parse(lst)))
        elif 'ingredients' in df.columns:
            df['ner_str'] = df['ingredients'].astype(str).str.lower()
        else:
            df['ner_str'] = ""
    df['ner_str'] = df['ner_str'].fillna('').astype(str)
    return df

def reduce_labels(y, top_k=30):
    cnt = Counter(y)
    most_common = set([c for c,_ in cnt.most_common(top_k)])
    return [c if c in most_common else "Other" for c in y]

def main(args):
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV file not found: {args.csv}")

    print("Loading CSV:", args.csv)
    df = prepare_dataframe(args.csv)
    if 'cuisine' not in df.columns or df['cuisine'].isna().all():
        print("No cuisine column found or all values missing. Add a 'cuisine' column to train.")
        return

    X = df['ner_str'].astype(str)
    y_raw = df['cuisine'].fillna('Unknown').astype(str)
    print("Original cuisine label counts:", Counter(y_raw).most_common(10))

    y = reduce_labels(y_raw, top_k=args.top_k)
    print("Reduced labels counts (top_k={}):".format(args.top_k), Counter(y).most_common(15))

    # simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    print("Train/test sizes:", len(X_train), len(X_test))

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    Xv_train = vectorizer.fit_transform(X_train)
    Xv_test  = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, multi_class='auto')
    print("Training classifier...")
    clf.fit(Xv_train, y_train)

    print("Evaluating on test set...")
    preds = clf.predict(Xv_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds, zero_division=0))

    # persist model + vectorizer
    model_data = {'model': clf, 'vectorizer': vectorizer}
    out_path = args.out
    with open(out_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Saved model to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="full_dataset.csv", help="Path to CSV dataset")
    p.add_argument("--out", type=str, default="cuisine_model.pkl", help="Output pickle path")
    p.add_argument("--top-k", type=int, default=30, help="Keep top-k most frequent cuisines; rest => 'Other'")
    args = p.parse_args()
    main(args)
