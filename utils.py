# utils.py
import re
import ast
import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

# ---------- Data loading ----------
def load_data(path="full_dataset.csv"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Put your CSV in the project folder or change DATA_PATH.")
    df = pd.read_csv(path)
    # Parse NER if present as stringified list
    if 'NER' in df.columns:
        def try_parse(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            try:
                v = ast.literal_eval(x)
                if isinstance(v, list):
                    return [str(i).strip() for i in v if i]
            except Exception:
                return [i.strip() for i in str(x).split(',') if i.strip()]
            return []
        df['NER'] = df['NER'].apply(try_parse)
    else:
        if 'ingredients' in df.columns:
            df['NER'] = df['ingredients'].apply(lambda s: [i.strip() for i in str(s).split(',') if i.strip()])
        else:
            df['NER'] = [[] for _ in range(len(df))]

    df['ner_lower'] = df['NER'].apply(lambda lst: [i.lower().strip() for i in lst if i])
    df['ner_unique'] = df['ner_lower'].apply(lambda lst: list(dict.fromkeys(lst)))
    df['ner_str'] = df['ner_unique'].apply(lambda lst: " ".join(lst))
    # normalize directions
    if 'directions' in df.columns:
        def ensure_list(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            return [s.strip() for s in re.split(r'[.\n]+', str(x)) if s.strip()]
        df['directions'] = df['directions'].apply(ensure_list)
    else:
        df['directions'] = [[] for _ in range(len(df))]
    if 'title' not in df.columns:
        df['title'] = ""
    return df

# ---------- normalization ----------
def normalize_ingredient(s: str):
    s = str(s).lower()
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'[^a-z0-9\s/-]', ' ', s)
    s = re.sub(r'\b(of|and|to|in|fresh|chopped|minced|sliced)\b', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------- searching & filters ----------
def search_by_ingredients(df, user_ingredients):
    if not user_ingredients:
        return df
    user_ingredients = [normalize_ingredient(x) for x in user_ingredients]
    def matches(row):
        row_ings = row.get('ner_unique', [])
        row_ings = [normalize_ingredient(x) for x in row_ings]
        return all(any(ui in ing for ing in row_ings) for ui in user_ingredients)
    return df[df.apply(matches, axis=1)]

def get_meal_type(ingredients):
    nonveg_keywords = ['chicken', 'beef', 'fish', 'mutton', 'shrimp', 'egg', 'lamb', 'pork']
    vegan_exclude = ['milk', 'cheese', 'egg', 'yogurt', 'butter', 'paneer', 'honey']
    if any(any(nv in ing for nv in nonveg_keywords) for ing in ingredients):
        return 'Non-Veg'
    if any(any(vx in ing for vx in vegan_exclude) for ing in ingredients):
        return 'Vegetarian'
    return 'Vegan'

def get_flavor_type(ingredients):
    sweet_kw = ['sugar', 'honey', 'chocolate', 'maple', 'cinnamon', 'vanilla', 'cake', 'sweet']
    savory_kw = ['salt', 'pepper', 'garlic', 'onion', 'curry', 'chilli', 'tomato', 'cheese', 'savory']
    is_sweet = any(any(sk in ing for sk in sweet_kw) for ing in ingredients)
    is_savory = any(any(sv in ing for sv in savory_kw) for ing in ingredients)
    if is_sweet and not is_savory:
        return 'Sweet'
    elif is_savory and not is_sweet:
        return 'Savory'
    elif is_sweet and is_savory:
        return 'Both'
    else:
        return 'Other'

def get_meal_time(title, directions):
    meal_kw = {
        'Breakfast': ['breakfast', 'pancake', 'cereal', 'morning', 'omelet'],
        'Lunch': ['lunch', 'sandwich', 'salad'],
        'Dinner': ['dinner', 'rice', 'pasta', 'evening', 'stew'],
        'Snack': ['snack', 'cookie', 'chips', 'appetizer', 'biscuit']
    }
    title_directions = (title or "").lower() + " " + " ".join(directions).lower()
    for meal, kws in meal_kw.items():
        if any(kw in title_directions for kw in kws):
            return meal
    return 'Other'

def quick_time_estimate(directions):
    length = len(directions)
    if length < 5:
        return 'Quick'
    elif length < 10:
        return 'Medium'
    else:
        return 'Long'

def apply_filters(df, user_ings, diet, flavor, meal_time, prep_time, dish_name):
    result = df.copy()
    if user_ings:
        result = search_by_ingredients(result, user_ings)
    if diet != 'Any':
        result = result[result['ner_unique'].apply(get_meal_type) == diet]
    if flavor != 'Any':
        result = result[result['ner_unique'].apply(get_flavor_type) == flavor]
    if meal_time != 'Any':
        result = result[result.apply(lambda row: get_meal_time(row['title'], row['directions']), axis=1) == meal_time]
    if prep_time != 'Any':
        result = result[result['directions'].apply(quick_time_estimate) == prep_time]
    if dish_name:
        result = result[result['title'].str.contains(dish_name, case=False, na=False)]
    return result

# ---------- TF-IDF based recommendations ----------
def build_tfidf_recipe_matrix(df):
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    mat = vectorizer.fit_transform(df['ner_str'].fillna(''))
    return mat, vectorizer


def get_recipe_recommendations(recipe_idx, tfidf_matrix, df, topn=5):
    row_vec = tfidf_matrix[recipe_idx]
    sims = cosine_similarity(row_vec, tfidf_matrix).flatten()
    top_idxs = sims.argsort()[::-1][1:topn+1]
    return df.iloc[top_idxs]['title'].tolist()

# ---------- Simple co-occurrence based substitutes (no gensim) ----------
def get_substitutes(ingredient, df=None, topn=5):
    """
    Fallback substitute suggestion: use token overlap and global frequency of ingredients.
    Requires df (full dataframe) to compute frequency; if df is None, returns empty list.
    """
    if df is None:
        return []
    ing = normalize_ingredient(ingredient)
    # frequency of ingredients across recipes
    counter = Counter([normalize_ingredient(i) for lst in df['ner_unique'] for i in lst if i])
    # build candidate scores using token overlap and frequency
    candidates = []
    ing_tokens = set(ing.split())
    for cand in counter:
        if cand == ing:
            continue
        token_overlap = len(ing_tokens.intersection(cand.split()))
        if token_overlap > 0:
            score = (token_overlap, counter[cand])
            candidates.append((cand, score))
    # sort by overlap first then frequency
    candidates.sort(key=lambda x: (x[1][0], x[1][1]), reverse=True)
    return [c[0] for c in candidates[:topn]]

# ---------- summarization wrapper (safe) ----------
def summarize_steps_safe(row, max_length=100):
    try:
        from transformers import pipeline
        directions = row.get('directions', [])
        text = " ".join(directions[:10]) if isinstance(directions, list) else str(directions)
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", truncation=True)
        out = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return out[0]['summary_text']
    except Exception:
        directions = row.get('directions', [])
        if isinstance(directions, list) and directions:
            return " ".join(directions[:3])
        return str(row.get('directions'))[:300]
