#!/usr/bin/env python3
"""
evaluate.py — Evaluate model on test parquet file.

The test set contains 1,889 harmful Thai prompts across 6 risk areas.
All prompts are toxic → correct prediction = TOXIC (1).
Measures recall: how many toxic prompts does the model catch?
"""
import os, sys, json, pickle, re, time
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

def load_model():
    with open(str(MODEL_DIR / "tfidf_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(str(MODEL_DIR / "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    keywords = []
    kw_path = MODEL_DIR / "toxic_keywords.json"
    if kw_path.exists():
        with open(str(kw_path), "r", encoding="utf-8") as f:
            keywords = json.load(f)
    harm_kws, harm_cats = [], {}
    hi_path = MODEL_DIR / "harm_intent_keywords.json"
    if hi_path.exists():
        with open(str(hi_path), "r", encoding="utf-8") as f:
            data = json.load(f)
            harm_kws = data.get("all", [])
            harm_cats = data.get("categories", {})
    return model, vectorizer, keywords, harm_kws, harm_cats

def preprocess(text):
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def predict_batch(texts, model, vectorizer, keywords, harm_kws, harm_cats):
    """Predict a batch of texts."""
    try:
        from pythainlp.tokenize import word_tokenize
        tokenized = [" ".join(word_tokenize(preprocess(t), engine="newmm")) for t in texts]
    except ImportError:
        tokenized = [preprocess(t) for t in texts]

    X_tfidf = vectorizer.transform(tokenized)

    # Build 13 keyword features matching training
    cat_names = ['sexual', 'self_harm', 'privacy', 'illegal', 'info_hazard', 'cultural', 'misinfo']
    n_features = 6 + len(cat_names)
    kw_feats = np.zeros((len(texts), n_features))

    for i, text in enumerate(texts):
        clean = preprocess(text)
        wc = max(len(clean.split()), 1)
        tc = sum(1 for kw in keywords if kw in clean)
        kw_feats[i, 0] = tc
        kw_feats[i, 1] = tc / wc
        kw_feats[i, 2] = 1.0 if tc > 0 else 0.0

        hc = sum(1 for kw in harm_kws if kw in clean)
        kw_feats[i, 3] = hc
        kw_feats[i, 4] = hc / wc
        kw_feats[i, 5] = 1.0 if hc > 0 else 0.0

        for j, cat in enumerate(cat_names):
            if cat in harm_cats:
                kw_feats[i, 6 + j] = 1.0 if any(kw in clean for kw in harm_cats[cat]) else 0.0

    X = hstack([X_tfidf, csr_matrix(kw_feats)])

    preds = model.predict(X)
    probas = model.predict_proba(X)

    # Safety net
    for i, text in enumerate(texts):
        clean = preprocess(text)
        matched = [kw for kw in keywords if kw in clean] + [kw for kw in harm_kws if kw in clean]
        if matched and probas[i][1] < 0.6:
            boost = min(0.95, probas[i][1] + 0.3 * len(matched))
            probas[i] = [1.0 - boost, boost]
            preds[i] = 1

    return preds, probas

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Thai Harassment Detection — Test Evaluation           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Load test data
    test_path = DATA_DIR / "test.parquet"
    if not test_path.exists():
        print(f"[ERROR] {test_path} not found")
        sys.exit(1)

    df = pd.read_parquet(str(test_path))
    print(f"  Test file: {len(df):,} prompts")
    print(f"  Risk areas: {df['risk_area'].nunique()}")
    print(f"  All prompts are harmful → expected label = TOXIC\n")

    # Load model
    model, vectorizer, keywords, harm_kws, harm_cats = load_model()
    with open(str(MODEL_DIR / "config.json")) as f:
        config = json.load(f)
    print(f"  Model: {config['model_type']} | Train accuracy: {config['accuracy']*100:.1f}%")
    print(f"  Keywords: {len(keywords)} toxic + {len(harm_kws)} harm-intent\n")

    # Predict
    print("  Running predictions...")
    start = time.perf_counter()
    texts = df['prompt'].tolist()
    preds, probas = predict_batch(texts, model, vectorizer, keywords, harm_kws, harm_cats)
    elapsed = time.perf_counter() - start
    print(f"  Done in {elapsed:.2f}s ({len(texts)/elapsed:.0f} prompts/sec)\n")

    # All should be toxic (1) → recall
    correct = sum(1 for p in preds if p == 1)
    missed = sum(1 for p in preds if p == 0)
    recall = correct / len(preds) * 100

    print("=" * 60)
    print(f"  OVERALL RESULTS")
    print("=" * 60)
    print(f"  Total prompts:    {len(preds):,}")
    print(f"  Detected toxic:   {correct:,}  ✓")
    print(f"  Missed (FN):      {missed:,}  ✗")
    print(f"  Recall:           {recall:.1f}%")
    print(f"  Avg toxic prob:   {np.mean(probas[:, 1]):.3f}")
    print()

    # Breakdown by risk area
    print("  RECALL BY RISK AREA")
    print("  " + "-" * 56)
    df['pred'] = preds
    df['toxic_prob'] = probas[:, 1]

    for area, group in df.groupby('risk_area'):
        area_correct = (group['pred'] == 1).sum()
        area_total = len(group)
        area_recall = area_correct / area_total * 100
        avg_prob = group['toxic_prob'].mean()
        bar = "█" * int(area_recall / 5) + "░" * (20 - int(area_recall / 5))
        print(f"  {bar} {area_recall:5.1f}% ({area_correct:>3}/{area_total:>3}) | {area[:45]}")

    print()

    # Breakdown by types_of_harm
    print("  RECALL BY HARM TYPE (top 10)")
    print("  " + "-" * 56)
    for harm, group in df.groupby('types_of_harm'):
        if len(group) >= 10:
            h_correct = (group['pred'] == 1).sum()
            h_total = len(group)
            h_recall = h_correct / h_total * 100
            print(f"  {h_recall:5.1f}% ({h_correct:>3}/{h_total:>3}) | {harm[:50]}")

    print()

    # Show worst misses (false negatives with lowest toxic probability)
    missed_df = df[df['pred'] == 0].sort_values('toxic_prob')
    if len(missed_df) > 0:
        print(f"  WORST MISSES (top 10 false negatives)")
        print("  " + "-" * 56)
        for _, row in missed_df.head(10).iterrows():
            print(f"  P={row['toxic_prob']:.3f} | {row['prompt'][:55]}")

    print()
    print("=" * 60)

    # Save results
    out_path = DATA_DIR / "test_results.csv"
    df.to_csv(str(out_path), index=False, encoding='utf-8-sig')
    print(f"  Full results saved: {out_path}")

if __name__ == '__main__':
    main()
