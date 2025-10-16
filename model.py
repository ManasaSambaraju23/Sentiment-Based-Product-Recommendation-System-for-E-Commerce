# model.py
# Minimal model module using Logistic Regression + simple sentiment-based "user-based" recommendation
# Exposes:
#   - recommend_products(username, top_k=5) -> list of product names
#   - predict_sentiment(text) -> "Positive"/"Negative"/"Neutral"/"Unknown"

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# --- Config: small / fast settings for TF-IDF & LR ---
TFIDF_MAX_FEATURES = 8000
TFIDF_NGRAM = (1, 2)
LR_C = 1.0
RANDOM_STATE = 42
SENT_PIPE_PATH = "sentiment_pipeline_quick.joblib"  

# --- Load data (sample30.csv ) ---
DATA_PATH = "sample30.csv"

# Helpers to detect column names robustly
_text_candidates = ['review', 'combined_text', 'reviews_text', 'reviews_title', 'review_text', 'text']
_user_candidates = ['reviews_username', 'user', 'username']
_item_candidates = ['name', 'product_name', 'product', 'Name']

def _load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(path)
    return df

def _detect_columns(df):
    # detect item (product) column
    item_col = next((c for c in _item_candidates if c in df.columns), None)
    # detect user column
    user_col = next((c for c in _user_candidates if c in df.columns), None)
    # detect text column (review)
    text_col = next((c for c in _text_candidates if c in df.columns), None)
    # detect sentiment column if present
    sent_col = 'user_sentiment' if 'user_sentiment' in df.columns else None
    return item_col, user_col, text_col, sent_col

# Load dataset and detect columns once
_df = _load_dataset(DATA_PATH)
_item_col, _user_col, _text_col, _sent_col = _detect_columns(_df)

if _item_col is None or _user_col is None:
    raise RuntimeError(f"Could not detect required columns in {DATA_PATH}. Found columns: {_df.columns.tolist()}")

# Normalize internal column names
_df = _df.rename(columns={_item_col: 'name', _user_col: 'reviews_username', **({_text_col: 'review'} if _text_col else {})})

# If a text column was missing, create empty text to avoid failures 
if 'review' not in _df.columns:
    _df['review'] = ""

# If sentiment column missing but you have explicit labels in text, attempt to create mapping
if _sent_col is None and 'user_sentiment' not in _df.columns:
    # try to find textual sentiment column candidates
    # if none found, we will predict later and/or use heuristics
    _sent_col_candidates = ['sentiment', 'sentiment_label']
    found = next((c for c in _sent_col_candidates if c in _df.columns), None)
    if found:
        _df['user_sentiment'] = _df[found]
    else:
        # no sentiment labels available; we'll predict sentiment using model trained on titles/text if possible
        pass
else:
    if _sent_col is not None:
        _df = _df.rename(columns={_sent_col: 'user_sentiment'})

# If user_sentiment exists and is textual like 'positive'/'negative', map to ints
if 'user_sentiment' in _df.columns:
    # Map common textual labels to ints 1 pos, 0 neg, 2 neutral
    if _df['user_sentiment'].dtype == object:
        mapping = {'positive': 1, 'pos': 1, 'negative': 0, 'neg': 0, 'neutral': 2, 'neutrality': 2}
        _df['user_sentiment'] = _df['user_sentiment'].astype(str).str.lower().map(mapping).astype('Int64')
    # ensure numeric type where possible
    try:
        _df['user_sentiment'] = _df['user_sentiment'].astype('float').astype('Int64')
    except Exception:
        pass

# --- Train or load quick sentiment pipeline (Logistic Regression) ---
_sentiment_pipeline = None
if os.path.exists(SENT_PIPE_PATH):
    try:
        _sentiment_pipeline = joblib.load(SENT_PIPE_PATH)
    except Exception:
        _sentiment_pipeline = None

def _train_quick_pipeline(df=_df):
    """
    Train a quick TF-IDF + LogisticRegression pipeline if we have labeled sentiment data.
    Returns pipeline or None if not trainable.
    """
    # need a target column 'user_sentiment' with at least two classes
    if 'user_sentiment' not in df.columns:
        return None
    df_train = df.dropna(subset=['user_sentiment'])
    if df_train.shape[0] < 50:
        return None
    # Use text from 'review' if available; else fall back to 'name' or combined_text
    Xcol = 'review' if df_train['review'].str.strip().replace('', pd.NA).notna().sum() > 0 else None
    if Xcol is None:
        # fallback to product name or combined_text
        if 'combined_text' in df_train.columns:
            Xcol = 'combined_text'
        else:
            Xcol = 'name'  # still better than nothing
    X = df_train[Xcol].astype(str).fillna('')
    y = df_train['user_sentiment'].astype(int).values
    # convert to binary: map neutral (2) to pos or handle as separate class. We'll keep binary: treat neutral as positive (or map explicitly)
    # For simplicity map neutral->1
    y = np.where(y == 2, 1, y)  # neutral -> positive
    # quick split 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM, stop_words='english')),
        ('clf', LogisticRegression(C=LR_C, class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE))
    ])
    pipeline.fit(X_train, y_train)
    # save for reuse
    try:
        joblib.dump(pipeline, SENT_PIPE_PATH)
    except Exception:
        pass
    return pipeline

# Ensure _sentiment_pipeline is available
if _sentiment_pipeline is None:
    _sentiment_pipeline = _train_quick_pipeline(_df)

# --- Build product-level sentiment summary ---
# If user_sentiment exists, aggregate; else if pipeline exists, predict on all reviews and aggregate.
_product_sentiment_df = None

def _build_product_sentiment():
    global _product_sentiment_df
    df_local = _df.copy()
    if 'user_sentiment' in df_local.columns and df_local['user_sentiment'].notna().sum() > 0:
        # use existing labels
        labels = df_local.dropna(subset=['user_sentiment']).copy()
        labels['user_sentiment'] = labels['user_sentiment'].astype(int)
        # normalize neutral=2 -> positive (1) for percentage, or keep separate; we'll count >0 as positive
        labels['is_positive'] = (labels['user_sentiment'] >= 1).astype(int)
        agg = labels.groupby('name')['is_positive'].agg(['count','sum']).reset_index().rename(columns={'count':'total_reviews','sum':'positive_reviews'})
        agg['positive_percentage'] = 100.0 * agg['positive_reviews'] / agg['total_reviews']
        _product_sentiment_df = agg[['name','total_reviews','positive_reviews','positive_percentage']]
    elif _sentiment_pipeline is not None:
        # predict in batches for memory safety
        texts = df_local['review'].astype(str).fillna('')
        if texts.shape[0] == 0:
            _product_sentiment_df = pd.DataFrame(columns=['name','total_reviews','positive_reviews','positive_percentage'])
            return
        B = 5000
        preds = []
        for i in range(0, texts.shape[0], B):
            batch = texts.iloc[i:i+B].tolist()
            try:
                proba = _sentiment_pipeline.predict_proba(batch)[:,1]
                pred_pos = (proba >= 0.5).astype(int)
            except Exception:
                pred_pos = _sentiment_pipeline.predict(batch)
            preds.extend(pred_pos)
        df_local['pred_pos'] = preds
        agg = df_local.groupby('name')['pred_pos'].agg(['count','sum']).reset_index().rename(columns={'count':'total_reviews','sum':'positive_reviews'})
        agg['positive_percentage'] = 100.0 * agg['positive_reviews'] / agg['total_reviews']
        _product_sentiment_df = agg[['name','total_reviews','positive_reviews','positive_percentage']]
    else:
        # no labels or model: produce empty table
        _product_sentiment_df = pd.DataFrame(columns=['name','total_reviews','positive_reviews','positive_percentage'])

# build once at import
try:
    _build_product_sentiment()
except Exception:
    _product_sentiment_df = pd.DataFrame(columns=['name','total_reviews','positive_reviews','positive_percentage'])

# --- Public API functions ---

def predict_sentiment(text):
    """
    Predict sentiment label for a single text input.
    Returns "Positive", "Negative", "Neutral", or "Unknown".
    """
    if _sentiment_pipeline is None:
        return "Unknown"
    if not isinstance(text, str):
        text = str(text)
    try:
        # prefer predict_proba -> threshold 0.5
        if hasattr(_sentiment_pipeline, 'predict_proba'):
            p = _sentiment_pipeline.predict_proba([text])[0,1]
            return "Positive" if p >= 0.5 else "Negative"
        else:
            lab = _sentiment_pipeline.predict([text])[0]
            return "Positive" if int(lab) >= 1 else "Negative"
    except Exception:
        return "Unknown"

def recommend_products(username, top_k=5):
    """
    Recommend top_k product NAMES for the given username.
    Returns:
      - list of product names (strings) length up to top_k
    """
    df = _df
    if username not in df['reviews_username'].unique():
        return []

    # user's reviews
    urev = df[df['reviews_username'] == username].copy()

    # compute user's average sentiment
    user_sent_vals = []
    if 'user_sentiment' in urev.columns and urev['user_sentiment'].notna().sum() > 0:
        sval = urev['user_sentiment'].dropna().astype(int).values
        sval = np.where(sval == 2, 1, sval)  # map neutral->positive
        user_avg = float(np.mean(sval))
    elif _sentiment_pipeline is not None and urev.shape[0] > 0:
        # predict on their texts
        texts = urev['review'].astype(str).tolist()
        try:
            probs = _sentiment_pipeline.predict_proba(texts)[:,1]
            preds = (probs >= 0.5).astype(int)
            user_avg = float(preds.mean())
        except Exception:
            preds = _sentiment_pipeline.predict(texts)
            preds = np.where(np.array(preds)==2, 1, preds)
            user_avg = float(np.mean(preds))
    else:
        # fallback: neutral-ish
        user_avg = 1.0  # assume positive if no info

    # product sentiment summary
    ps = _product_sentiment_df.copy()
    if ps.empty:
        # fallback: recommend most reviewed products the user hasn't reviewed yet
        most_reviewed = df.groupby('name').size().reset_index(name='count').sort_values('count', ascending=False)
        seen = set(urev['name'].unique())
        recs = [n for n in most_reviewed['name'].tolist() if n not in seen][:top_k]
        return recs

    # compute closeness to user_avg
    ps['distance'] = (ps['positive_percentage']/100.0 - user_avg).abs()
    # exclude products user has already reviewed
    seen = set(urev['name'].unique())
    ps = ps[~ps['name'].isin(seen)].copy()
    # sort by distance ASC then positive_percentage DESC (prefer close & higher positivity)
    ps = ps.sort_values(by=['distance','positive_percentage'], ascending=[True, False])
    top = ps.head(top_k)['name'].tolist()
    return top

# quick CLI test when running model.py directly
if __name__ == "__main__":
    print("Loaded data with columns:", _df.columns.tolist())
    print("Sample product sentiment rows:")
    print(_product_sentiment_df.head())
    # sample username
    sample_users = _df['reviews_username'].dropna().unique().tolist()[:5]
    print("Sample users:", sample_users)
    if sample_users:
        print("Recommendations for", sample_users[0], "->", recommend_products(sample_users[0], top_k=5))
