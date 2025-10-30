# app.py
import streamlit as st
from utils import (
    load_data,
    normalize_ingredient,
    apply_filters,
    build_tfidf_recipe_matrix,
    get_recipe_recommendations,
    get_substitutes,
    summarize_steps_safe,
)
from nutrition import compute_nutrition_for_recipe

st.set_page_config(page_title="Smart Recipe Recommendation Engine", layout="wide")
st.title("🍽️ Smart Recipe Recommendation Engine")

DATA_PATH = "full_dataset.csv"  # <-- update to your CSV filename if different
try:
    df = load_data(DATA_PATH)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Build TF-IDF matrix once
tfidf_matrix, tfidf_vectorizer = build_tfidf_recipe_matrix(df)

with st.sidebar:
    st.header("Search options")
    search_mode = st.radio("Search mode", ["By ingredients", "By dish name"])
    if search_mode == "By ingredients":
        user_ings_raw = st.text_input("Enter ingredients (comma separated, up to 8):")
        user_ings = [normalize_ingredient(x) for x in user_ings_raw.split(",") if x.strip()][:8]
    else:
        dish_name = st.text_input("Enter dish name (substring match):")
        user_ings = []

    st.markdown("### Filters")
    diet = st.selectbox("Diet", ["Any", "Vegetarian", "Non-Veg", "Vegan"])
    flavor = st.selectbox("Flavor", ["Any", "Sweet", "Savory", "Both", "Other"])
    meal_time = st.selectbox("Meal type", ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Other"])
    prep_time = st.selectbox("Prep time", ["Any", "Quick", "Medium", "Long"])
    max_results = st.slider("Max results to show", 5, 50, 10)

    show_recommendations = st.checkbox("Show recipe recommendations for first match", value=True)
    show_nutrition = st.checkbox("Show nutrition info for first match", value=True)
    show_substitutes = st.checkbox("Show ingredient substitutes", value=True)
    show_summary = st.checkbox("Show summarized steps (may require transformers)", value=False)

    submitted = st.button("Search")

if submitted:
    if search_mode == "By dish name" and dish_name:
        filtered = df[df['title'].str.contains(dish_name, case=False, na=False)]
    else:
        filtered = apply_filters(df, user_ings, diet, flavor, meal_time, prep_time, dish_name="")

    st.subheader(f"Found {len(filtered)} recipes")
    if len(filtered) == 0:
        st.info("No recipes found. Try fewer ingredient constraints or 'Any' filters.")
    else:
        for idx, row in filtered.head(max_results).iterrows():
            st.markdown(f"### {row.get('title','(No title)')}")
            cols = st.columns([2, 1])
            with cols[0]:
                st.write("**Ingredients (NER / extracted)**")
                ings = row.get("ner_unique") or row.get("ner_lower") or []
                st.write(", ".join(ings[:50]))
                st.write("**Directions (first lines)**")
                directions = row.get("directions") or []
                if isinstance(directions, list):
                    st.write("\n".join(directions[:6]))
                else:
                    st.write(str(directions)[:500] + "...")

            with cols[1]:
                if show_recommendations:
                    if st.button(f"Recommend similar for #{idx}", key=f"rec{idx}"):
                        recs = get_recipe_recommendations(idx, tfidf_matrix, df, topn=5)
                        st.write("**Similar recipes:**")
                        for r in recs:
                            st.write(f"- {r}")

                if show_nutrition:
                    nutr = compute_nutrition_for_recipe(row)
                    st.write("**Nutrition (approx)**")
                    cal = nutr.get('calories')
                    st.metric("Calories", f"{cal:.0f}" if cal else "N/A")
                    st.write(nutr)

            if show_substitutes:
                st.write("**Ingredient substitutes (top 4 each):**")
                ing_list = (row.get("ner_unique") or row.get("ner_lower") or [])[:8]
                for ing in ing_list:
                    subs = get_substitutes(ing, df=df, topn=4)
                    st.write(f"- {ing}: {', '.join(subs) if subs else 'None found'}")

            if show_summary:
                try:
                    s = summarize_steps_safe(row)
                    st.write("**Summary of steps:**")
                    st.write(s)
                except Exception as e:
                    st.error(f"Summarization failed: {e}")

            st.write("---")

else:
    st.info("Enter search options in the sidebar and click Search.")
