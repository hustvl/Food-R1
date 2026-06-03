"""Food-R1 GRPO reward functions for ms-swift.

Use with:
    swift rlhf ... --plugin /path/to/Food-R1/foodr1/rewards.py \
        --reward_funcs ingredient_format ingredient_match ingredient_quantity_match total_kcal_exact
"""

import re
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

try:
    from swift.plugin import ORM, orms
except Exception:  # Allows local syntax/import checks without ms-swift installed.
    class ORM:
        def __call__(self, *args, **kwargs):
            raise NotImplementedError

    orms = {}


DATASET_CALORIEBENCH80K = "caloriebench_80k"
DATASET_NUTRITION5K = "nutrition5k"
DATASET_RECIPE1M = "recipe1m"
DATASET_VIREO172 = "vireo172"


def extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    return text.strip()


def clean(text: str) -> str:
    if not text:
        return ""
    return (
        text.replace("<|endoftext|>", "")
        .replace("ingradient", "ingredient")
        .replace("ingradients", "ingredients")
        .strip(" .,;:\n\t")
        .lower()
    )


_UNIT_NORM = {
    "g": "g",
    "gram": "g",
    "grams": "g",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "mg": "mg",
    "milligram": "mg",
    "milligrams": "mg",
    "ml": "ml",
    "milliliter": "ml",
    "milliliters": "ml",
    "l": "l",
    "liter": "l",
    "liters": "l",
    "oz": "oz",
    "ounce": "oz",
    "ounces": "oz",
    "lb": "lb",
    "lbs": "lb",
    "pound": "lb",
    "pounds": "lb",
    "tsp": "tsp",
    "teaspoon": "tsp",
    "teaspoons": "tsp",
    "tbsp": "tbsp",
    "tablespoon": "tbsp",
    "tablespoons": "tbsp",
    "cup": "cup",
    "cups": "cup",
    "slice": "slice",
    "slices": "slice",
    "clove": "clove",
    "cloves": "clove",
    "piece": "piece",
    "pieces": "piece",
    "pinch": "pinch",
    "pinches": "pinch",
    "dash": "dash",
    "dashes": "dash",
    "handful": "handful",
    "handfuls": "handful",
    "%": "%",
}


def norm_unit(unit: str) -> str:
    if not unit:
        return ""
    unit = clean(unit)
    return _UNIT_NORM.get(unit, unit)


_RE_ITEM_1 = re.compile(
    r"^\s*(?P<qty>[\d\.\s/]+)\s*(?P<unit>[a-zA-Z%]+)?\s*(?:of\s+)?(?P<name>.+?)\s*$",
    re.I,
)
_RE_ITEM_2 = re.compile(
    r"^\s*(?P<qty>\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+)(?P<unit>[a-zA-Z%]+)\s*(?:of\s+)?(?P<name>.+?)\s*$",
    re.I,
)


def _parse_item(text: str) -> Tuple[str, str, str]:
    item = text.strip().strip(".")
    if item.lower().startswith("and "):
        item = item[4:].strip()
    match = _RE_ITEM_1.match(item) or _RE_ITEM_2.match(item)
    if match:
        qty = match.group("qty").strip() if match.group("qty") else ""
        unit = norm_unit(match.group("unit") or "")
        name = clean(match.group("name"))
        return (qty, unit, name) if name else ("", "", "")
    return "", "", clean(item)


def _split_items(tail: str) -> List[str]:
    parts = [part.strip() for part in re.split(r",\s*", tail) if part.strip()]
    if parts and parts[-1].lower().startswith("and "):
        parts[-1] = parts[-1][4:].strip()
    return parts


def extract_ingredients_from_they_are(text: str) -> List[Tuple[str, str, str]]:
    ans = extract_answer(text)
    match = re.search(
        r"\bthey are\b\s*:\s*(.*?)\.\s*(?:they have a total\b|$)",
        ans,
        flags=re.I | re.S,
    )
    if not match:
        match = re.search(
            r"\bthey are\b\s*(.*?)\.\s*(?:they have a total\b|$)",
            ans,
            flags=re.I | re.S,
        )
    if not match:
        return []

    parsed = []
    for item in _split_items(match.group(1).strip()):
        qty, unit, name = _parse_item(item)
        if qty or unit or name:
            parsed.append((qty, unit, name))
    return parsed


def _normalize_qty(qty: str) -> Optional[float]:
    if not qty:
        return None
    try:
        return float(eval(qty.replace(" ", "+"), {"__builtins__": {}}, {}))
    except Exception:
        return None


def _is_caloriebench80k_row(index: int, dataset) -> bool:
    return dataset is None or (index < len(dataset) and dataset[index] == DATASET_CALORIEBENCH80K)


class IngredientFormatReward(ORM):
    _HEAD = re.compile(
        r"^\s*The dish is (?P<name>[^,]+), and has (?P<n>\d+)\s+ing(?:redients|radients?) in total\.\s*"
        r"They are:\s*(?P<tail>.+?)\.\s*"
        r"They have a total of\s+(?P<kcal>\d+(?:\.\d+)?)\s*kcal\.\s*$",
        re.I | re.S,
    )

    def __call__(self, completions, solution=None, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, raw in enumerate(completions):
            if not _is_caloriebench80k_row(i, dataset):
                rewards.append(0.0)
                continue

            m_think = re.search(r"<think>\s*(.*?)\s*</think>", raw, flags=re.S | re.I)
            m_ans = re.search(r"<answer>(.*?)</answer>", raw, flags=re.S | re.I)
            if not m_think or not m_ans or m_think.start() > m_ans.start():
                rewards.append(0.0)
                continue
            if not re.search(r"\S", m_think.group(1) or ""):
                rewards.append(0.0)
                continue

            text = m_ans.group(1).replace("\n", " ").strip()
            score = 0.0
            match = self._HEAD.match(text)
            if match:
                items = _split_items(match.group("tail").strip())
                parsed = [_parse_item(item) for item in items]
                ok_shape = parsed and all(name or qty or unit for qty, unit, name in parsed)
                if ok_shape:
                    try:
                        declared = int(match.group("n"))
                        observed = len(items)
                        score = min(declared, observed) / max(declared, observed, 1)
                    except Exception:
                        score = 0.0
            rewards.append(float(score))
        return rewards


class IngredientMatchReward(ORM):
    """Jaccard reward over ingredient-name sets."""

    def __call__(self, completions, solution, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, (pred_text, sol_text) in enumerate(zip(completions, solution)):
            if not _is_caloriebench80k_row(i, dataset):
                rewards.append(0.0)
                continue

            pred_names = {name for _, _, name in extract_ingredients_from_they_are(pred_text) if name}
            sol_names = {name for _, _, name in extract_ingredients_from_they_are(sol_text) if name}
            if not pred_names and not sol_names:
                rewards.append(1.0)
                continue
            union = len(pred_names | sol_names)
            rewards.append((len(pred_names & sol_names) / union) if union else 0.0)
        return rewards


class IngredientQuantityMatchReward(ORM):
    """Soft quantity reward for exact name+unit matches."""

    def __call__(self, completions, solution, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, (pred_text, sol_text) in enumerate(zip(completions, solution)):
            if not _is_caloriebench80k_row(i, dataset):
                rewards.append(0.0)
                continue

            pred_list = [
                (name, norm_unit(unit), _normalize_qty(qty))
                for qty, unit, name in extract_ingredients_from_they_are(pred_text)
                if name and _normalize_qty(qty) is not None
            ]
            sol_list = [
                (name, norm_unit(unit), _normalize_qty(qty))
                for qty, unit, name in extract_ingredients_from_they_are(sol_text)
                if name and _normalize_qty(qty) is not None
            ]

            used = set()
            total = 0.0
            matched = 0
            for pred_name, pred_unit, pred_qty in pred_list:
                match_index = -1
                for j, (sol_name, sol_unit, sol_qty) in enumerate(sol_list):
                    if j in used:
                        continue
                    if pred_name == sol_name and pred_unit == sol_unit:
                        match_index = j
                        break
                if match_index >= 0:
                    _, _, sol_qty = sol_list[match_index]
                    diff = abs(pred_qty - sol_qty)
                    total += float(2.0 * np.exp(-diff) / (1.0 + np.exp(-diff)))
                    matched += 1
                    used.add(match_index)

            rewards.append(total / matched if matched else 0.0)
        return rewards


def _extract_total_kcal(text: str) -> Optional[float]:
    ans = extract_answer(text)
    match = re.search(r"they have a total of\s*([0-9]+(?:\.[0-9]+)?)\s*kcal\b", ans, flags=re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


class TotalKcalExactMatchReward(ORM):
    """Exact-match reward over total kcal."""

    def __call__(self, completions, solution, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, (pred_text, sol_text) in enumerate(zip(completions, solution)):
            if not _is_caloriebench80k_row(i, dataset):
                rewards.append(0.0)
                continue
            pred_kcal = _extract_total_kcal(pred_text)
            sol_kcal = _extract_total_kcal(sol_text)
            rewards.append(1.0 if pred_kcal is not None and pred_kcal == sol_kcal else 0.0)
        return rewards


def _get_dataset_value(dataset, index):
    if dataset is None:
        return None
    if isinstance(dataset, (str, dict)):
        return dataset
    try:
        return dataset[index]
    except Exception:
        return None


def _dataset_contains(dataset_value, token: str) -> bool:
    if dataset_value is None:
        return False
    return token.lower() in str(dataset_value).lower()


def _has_valid_think_answer_blocks(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if not re.match(r"^<think>.*?</think>\s*<answer>.*?</answer>\s*$", text, flags=re.S | re.I):
        return False
    m_think = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.S | re.I)
    m_answer = re.search(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if not m_think or not m_answer:
        return False
    return m_think.start() < m_answer.start() and bool(re.search(r"\S", m_think.group(1) or ""))


# ---------------------------------------------------------------------
# VireoFood-172 ingredient rewards. Only format and ingredient-set
# matching objectives are registered for release.
# ---------------------------------------------------------------------


def _is_vireo172(dataset_value) -> bool:
    return _dataset_contains(dataset_value, DATASET_VIREO172) or _dataset_contains(dataset_value, "vireofood172")


def _extract_vireo_ingredients(text: str):
    ans = extract_answer(text)
    match = re.search(r"\bthey are\b\s*:\s*(.*?)\.\s*$", ans, flags=re.I | re.S)
    if not match:
        match = re.search(r"\bthey are\b\s*:\s*(.*)$", ans, flags=re.I | re.S)
    if not match:
        return []
    items = [item.strip() for item in re.split(r",\s*", match.group(1).strip()) if item.strip()]
    if items and items[-1].lower().startswith("and "):
        items[-1] = items[-1][4:].strip()
    return [clean(item) for item in items if clean(item)]


class Vireo172FormatReward(ORM):
    _ANSWER_PATTERN = re.compile(
        r"^\s*The dish is\s+(?P<dish>.+?)\.\s*"
        r"It has\s+(?P<n>\d+)\s+ingredients?\s+in total\.\s*"
        r"They are\s*:\s*(?P<tail>.+?)\.\s*$",
        re.I | re.S,
    )

    def __call__(self, completions, solution=None, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, raw in enumerate(completions):
            ds = _get_dataset_value(dataset, i)
            if dataset is not None and not _is_vireo172(ds):
                rewards.append(0.0)
                continue
            if not _has_valid_think_answer_blocks(raw):
                rewards.append(0.0)
                continue
            answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.S | re.I)
            body = answer_match.group(1).replace("\n", " ").strip() if answer_match else ""
            match = self._ANSWER_PATTERN.match(body)
            if not match:
                rewards.append(0.0)
                continue
            items = [item.strip() for item in re.split(r",\s*", match.group("tail").strip()) if item.strip()]
            if items and items[-1].lower().startswith("and "):
                items[-1] = items[-1][4:].strip()
            if not items:
                rewards.append(0.0)
                continue
            try:
                declared = int(match.group("n"))
                rewards.append(min(declared, len(items)) / max(declared, len(items), 1))
            except Exception:
                rewards.append(0.0)
        return rewards


class Vireo172IngredientMatchReward(ORM):
    def __call__(self, completions, solution, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, (pred_text, gt_text) in enumerate(zip(completions, solution)):
            ds = _get_dataset_value(dataset, i)
            if dataset is not None and not _is_vireo172(ds):
                rewards.append(0.0)
                continue
            pred = set(_extract_vireo_ingredients(pred_text))
            gt = set(_extract_vireo_ingredients(gt_text))
            if not pred and not gt:
                rewards.append(1.0)
                continue
            union = len(pred | gt)
            rewards.append((len(pred & gt) / union) if union else 0.0)
        return rewards


# ---------------------------------------------------------------------
# Nutrition5k rewards.
# ---------------------------------------------------------------------


_NUM = r"[-+]?\d+(?:\.\d+)?"
_RE_N5K_ING_CAL = re.compile(
    rf"^The ingredient is (?P<name>.+?)\.\s*It weighs (?P<mass>{_NUM})\s*g\s*and has about (?P<kcal>{_NUM})\s*kcal in total\.\s*$",
    re.I | re.S,
)
_RE_N5K_ING_NUT = re.compile(
    rf"^The ingredient is (?P<name>.+?)\.\s*It weighs (?P<mass>{_NUM})\s*g\s*and provides about (?P<kcal>{_NUM})\s*kcal in total,\s*"
    rf"including (?P<fat>{_NUM})\s*g of fat,\s*(?P<carb>{_NUM})\s*g of carbohydrate,\s*and (?P<protein>{_NUM})\s*g of protein\.\s*$",
    re.I | re.S,
)
_RE_N5K_TOTAL_NUT = re.compile(
    rf"^The dish weighs (?P<mass>{_NUM})\s*g in total and provides about (?P<kcal>{_NUM})\s*kcal,\s*"
    rf"including (?P<fat>{_NUM})\s*g of fat,\s*(?P<carb>{_NUM})\s*g of carbohydrate,\s*and (?P<protein>{_NUM})\s*g of protein overall\.\s*$",
    re.I | re.S,
)


def _to_float_safe(value):
    try:
        return float(value)
    except Exception:
        return None


def _eq_intpart(pred, gt) -> bool:
    try:
        return int(float(pred)) == int(float(gt))
    except Exception:
        return False


def _is_nutrition5k(dataset_value) -> bool:
    return _dataset_contains(dataset_value, DATASET_NUTRITION5K)


def _n5k_infer_type_and_fields(text: str):
    ans = extract_answer(text).strip().replace("\n", " ")
    for task_type, pattern in (
        ("ingredient_calories", _RE_N5K_ING_CAL),
        ("ingredient_nutrition", _RE_N5K_ING_NUT),
        ("total_nutrition", _RE_N5K_TOTAL_NUT),
    ):
        match = pattern.match(ans)
        if match:
            fields = {key: _to_float_safe(value) for key, value in match.groupdict().items() if key != "name"}
            if "name" in match.groupdict():
                fields["name"] = match.group("name").strip()
            return task_type, fields
    return None, None


def _n5k_parse_pred_fields_by_type(text: str, task_type: str):
    ans = extract_answer(text).strip().replace("\n", " ")
    pattern = {
        "ingredient_calories": _RE_N5K_ING_CAL,
        "ingredient_nutrition": _RE_N5K_ING_NUT,
        "total_nutrition": _RE_N5K_TOTAL_NUT,
    }.get(task_type)
    if pattern is None:
        return None
    match = pattern.match(ans)
    if not match:
        return None
    fields = {key: _to_float_safe(value) for key, value in match.groupdict().items() if key != "name"}
    if "name" in match.groupdict():
        fields["name"] = match.group("name").strip()
    return fields


class Nutrition5kFormatReward(ORM):
    def __call__(self, completions, solution, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, (pred_text, gt_text) in enumerate(zip(completions, solution)):
            ds = _get_dataset_value(dataset, i)
            if dataset is not None and not _is_nutrition5k(ds):
                rewards.append(0.0)
                continue
            task_type, _ = _n5k_infer_type_and_fields(gt_text)
            if task_type is None or not _has_valid_think_answer_blocks(pred_text):
                rewards.append(0.0)
                continue
            rewards.append(1.0 if _n5k_parse_pred_fields_by_type(pred_text, task_type) is not None else 0.0)
        return rewards


class Nutrition5kKcalAccuracyReward(ORM):
    def __call__(self, completions, solution, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, (pred_text, gt_text) in enumerate(zip(completions, solution)):
            ds = _get_dataset_value(dataset, i)
            if dataset is not None and not _is_nutrition5k(ds):
                rewards.append(0.0)
                continue
            task_type, gt_fields = _n5k_infer_type_and_fields(gt_text)
            if task_type != "ingredient_calories" or gt_fields is None:
                rewards.append(0.0)
                continue
            pred_fields = _n5k_parse_pred_fields_by_type(pred_text, task_type)
            if pred_fields is None:
                rewards.append(0.0)
                continue
            rewards.append(1.0 if _eq_intpart(pred_fields.get("mass"), gt_fields.get("mass")) and _eq_intpart(pred_fields.get("kcal"), gt_fields.get("kcal")) else 0.0)
        return rewards


class Nutrition5kFullNutritionAccuracyReward(ORM):
    def __call__(self, completions, solution, dataset=None, **kwargs) -> List[float]:
        rewards = []
        for i, (pred_text, gt_text) in enumerate(zip(completions, solution)):
            ds = _get_dataset_value(dataset, i)
            if dataset is not None and not _is_nutrition5k(ds):
                rewards.append(0.0)
                continue
            task_type, gt_fields = _n5k_infer_type_and_fields(gt_text)
            if task_type not in ("ingredient_nutrition", "total_nutrition") or gt_fields is None:
                rewards.append(0.0)
                continue
            pred_fields = _n5k_parse_pred_fields_by_type(pred_text, task_type)
            if pred_fields is None:
                rewards.append(0.0)
                continue
            keys = ["mass", "kcal", "fat", "carb", "protein"]
            rewards.append(1.0 if all(_eq_intpart(pred_fields.get(key), gt_fields.get(key)) for key in keys) else 0.0)
        return rewards



orms["ingredient_format"] = IngredientFormatReward
orms["ingredient_match"] = IngredientMatchReward
orms["ingredient_quantity_match"] = IngredientQuantityMatchReward
orms["total_kcal_exact"] = TotalKcalExactMatchReward
orms["vireo172_format"] = Vireo172FormatReward
orms["vireo172_ingredient_match"] = Vireo172IngredientMatchReward
orms["nutrition5k_format"] = Nutrition5kFormatReward
orms["nutrition5k_kcal_accuracy"] = Nutrition5kKcalAccuracyReward
orms["nutrition5k_full_nutrition_accuracy"] = Nutrition5kFullNutritionAccuracyReward
