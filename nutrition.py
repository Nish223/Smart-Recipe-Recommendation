# nutrition.py
# Minimal example nutrition mapping. Replace/expand with a full CSV or database for production.

NUTRITION_DB = {
    "salt": {"calories": 0, "protein": 0, "fat": 0},
    "sugar": {"calories": 387, "protein": 0, "fat": 0},
    "olive oil": {"calories": 884, "protein": 0, "fat": 100},
    "chicken": {"calories": 239, "protein": 27, "fat": 14},
    "rice": {"calories": 130, "protein": 2.7, "fat": 0.3},
    "egg": {"calories": 155, "protein": 13, "fat": 11},
    "milk": {"calories": 42, "protein": 3.4, "fat": 1},
    # add more...
}

def lookup_ingredient_nutrition(ing):
    ing = ing.lower()
    # direct match
    if ing in NUTRITION_DB:
        return NUTRITION_DB[ing]
    # try token match
    for key in NUTRITION_DB:
        if key in ing:
            return NUTRITION_DB[key]
    return {}

def compute_nutrition_for_recipe(row):
    ings = row.get('ner_unique', []) or []
    total = {"calories": 0.0, "protein": 0.0, "fat": 0.0}
    for ing in ings:
        info = lookup_ingredient_nutrition(ing)
        if info:
            total['calories'] += info.get('calories', 0)
            total['protein'] += info.get('protein', 0)
            total['fat'] += info.get('fat', 0)
    return total
