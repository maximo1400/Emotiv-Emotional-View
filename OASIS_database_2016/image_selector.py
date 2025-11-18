import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from pathlib import Path
from math import ceil
from PIL import Image as PILImage
import pickle


import matplotlib.pyplot as plt

# fmt: off
# Converted lists for categories, plus som hardcoded ones for easy access
high_valence = ["Dog 6","Lake 9","Rainbow 2","Sunset 3",]
low_valence = ["Miserable pose 3","Tumor 1","Fire 9","Cockroach 1",]
high_arousal = ["Explosion 5","Parachuting 4","Snake 4","Lava 1",]
low_arousal = ["Wall 2","Cotton swabs 3","Office supplies 2","Socks 1",]
neutral = []

oasis_categories = {
    "high_valence": high_valence,
    "low_valence": low_valence,
    "high_arousal": high_arousal,
    "low_arousal": low_arousal,
    "neutral": neutral,
}

oasis_csv_path = "OASIS_database_2016/OASIS.csv"
out_pickle_path = "img_selector_output.pkl"
out_img_folder = "img"
img_per_category = 10
percentile = 0.1
banned_words = ["nude", "war"]
# fmt: on


def load_oasis(csv_path: str = oasis_csv_path) -> pd.DataFrame:
    """Load `csv_path` into a pandas DataFrame (strings, no NA filtering)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(path)
    first_col = df.columns[0]
    rename_map = {}
    rename_map[first_col] = "id"
    rename_map["Theme"] = "img"
    df = df.rename(columns=rename_map)
    if "id" in df.columns:
        df = df.set_index("id")
    return df


def sample_images_by_valence_arousal(
    df: pd.DataFrame,
    valence_group: str = "high",
    arousal_group: str = "high",
    n: int = 5,
    mode: str = "both",
    percentile_p: float = 0.33,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Sample images by valence/arousal groups.

    Parameters
    - df: DataFrame with columns ['img', 'Valence_mean', 'Arousal_mean'].
    - valence_group / arousal_group: 'high'|'low'|'mid'|'all'.
    - n: number of unique images to return (returns all rows for each sampled image).
    - mode: 'both' (default) | 'valence' | 'arousal' — which axis to apply filtering on.
    - percentile_p: percentile fraction for low/high split (e.g. 0.33 = bottom/top third).
    - valence_thresh, arousal_thresh: when method=='thresholds', provide (low_cut, high_cut).
    - random_state: seed for reproducible sampling.

    Returns
    - DataFrame with rows for the sampled images (all rows per image).
    """
    required = {"img", "Valence_mean", "Arousal_mean"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    for g in (valence_group, arousal_group):
        if g not in {"high", "low", "mid", "all"}:
            raise ValueError("group must be one of 'high','low','mid','all'")

    if mode not in {"both", "valence", "arousal"}:
        raise ValueError("mode must be 'both', 'valence' or 'arousal'")

    center_width = 2 * percentile_p

    # allow pair or single value for center_width
    if isinstance(center_width, tuple) or isinstance(center_width, list):
        v_center_w, a_center_w = float(center_width[0]), float(center_width[1])
    else:
        v_center_w = a_center_w = float(center_width)

    # compute thresholds
    v_low_q = df["Valence_mean"].quantile(percentile_p)
    v_high_q = df["Valence_mean"].quantile(1 - percentile_p)
    a_low_q = df["Arousal_mean"].quantile(percentile_p)
    a_high_q = df["Arousal_mean"].quantile(1 - percentile_p)

    def group_mask(series: pd.Series, group: str, low_q: float, high_q: float, center_w: float) -> pd.Series:
        if group == "all":
            return pd.Series(True, index=series.index)
        if group == "low":
            return series <= low_q
        if group == "high":
            return series >= high_q
        if group == "mid":
            half = max(0.0, min(0.5, center_w / 2.0))
            low = series.quantile(0.5 - half)
            high = series.quantile(0.5 + half)
            return (series >= low) & (series <= high)
        # fallback
        return (series > low_q) & (series < high_q)

    # build masks depending on mode
    if mode == "both":
        v_mask = group_mask(df["Valence_mean"], valence_group, v_low_q, v_high_q, v_center_w)
        a_mask = group_mask(df["Arousal_mean"], arousal_group, a_low_q, a_high_q, a_center_w)
        combined_mask = v_mask & a_mask
    elif mode == "valence":
        combined_mask = group_mask(df["Valence_mean"], valence_group, v_low_q, v_high_q, v_center_w)
    else:  # mode == "arousal"
        combined_mask = group_mask(df["Arousal_mean"], arousal_group, a_low_q, a_high_q, a_center_w)

    candidates = df.loc[combined_mask, "img"].unique()
    if len(candidates) == 0:
        raise ValueError("No candidate images found for the requested groups/mode/thresholds.")

    rng = np.random.RandomState(random_state)
    k = min(int(n), len(candidates))
    sampled_imgs = rng.choice(candidates, size=k, replace=False)

    return df[df["img"].isin(sampled_imgs)].reset_index(drop=True)


def image_selector(
    categories: Dict,
    img_per_category: int = 10,
    percentile: float = 0.1,
    banned_words: list = banned_words,
    oasis_df: pd.DataFrame = None,
):
    for cat in categories.keys():
        # print(f"Category: {cat}")
        len_img = len(categories[cat])
        while len_img < img_per_category:
            diff = img_per_category - len_img
            if cat == "neutral":
                group = "mid"
                mode = "both"
            else:
                group, mode = cat.split("_")

            # print(f"  Sampling {diff} images for group '{group}' in mode '{mode}' with percentile {percentile}")
            df = sample_images_by_valence_arousal(
                mode=mode, valence_group=group, arousal_group=group, n=diff, percentile_p=percentile, df=oasis_df
            )
            categories[cat].extend(df["img"].tolist())

            # remove banned items starting with any banned word (case-insensitive)
            bw_lower = [bw.lower() for bw in banned_words]
            before = len(categories[cat])
            categories[cat] = [
                name for name in categories[cat] if not any(name.lower().startswith(bw) for bw in bw_lower)
            ]
            removed = before - len(categories[cat])
            # if removed:
            #     print(f"  Removed {removed} banned item(s)")

            categories[cat] = list(set(categories[cat]))
            len_img = len(categories[cat])


def get_oasis_record(image_names, df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dict mapping each requested name to a small record dict (or None if not found).

    - `image_names` must be an iterable of names (list/tuple/Series). A single string is NOT accepted.
    - Returns: {name: {"img": ..., "Valence_mean": ..., "Arousal_mean": ...} | None}
    """
    if isinstance(image_names, str) or not hasattr(image_names, "__iter__"):
        raise TypeError("image_names must be an iterable of strings (e.g. list/tuple), not a single string")

    df_str = df.astype(str)
    names = [str(n) for n in list(image_names)]
    result: Dict[str, Any] = {}

    for name in names:
        mask = df_str["img"] == name

        matches = df.loc[mask]
        if matches.empty:
            print(f"Warning: no match found for image name '{name}'")
        else:
            m = matches.iloc[0].to_dict()
            result[name] = {
                "Valence_mean": m.get("Valence_mean"),
                "Arousal_mean": m.get("Arousal_mean"),
            }

    return result


def categorize_selected_imgs(
    image_names: list,
    df: pd.DataFrame,
    percentile_p: float = 0.33,
) -> Dict[str, Any]:
    """Categorize a list of images by valence/arousal groups.

    Parameters
    - image_names: iterable of image base names (list/tuple/Series). A single string is NOT accepted.
    - df: oasis dataframe
    - percentile_p: percentile fraction for low/high split (e.g. 0.33 = bottom/top third)

    Returns:
    - dict mapping image_name -> {img, Valence_mean, Arousal_mean, valence_group, arousal_group} or None if not found
    """
    if isinstance(image_names, str) or not hasattr(image_names, "__iter__"):
        raise TypeError("image_names must be an iterable of strings (e.g. list/tuple), not a single string")

    # get records for all requested names (get_oasis_record already warns for missing names)
    records = get_oasis_record(list(image_names), df)

    v_low_q = df["Valence_mean"].quantile(percentile_p)
    v_high_q = df["Valence_mean"].quantile(1 - percentile_p)
    a_low_q = df["Arousal_mean"].quantile(percentile_p)
    a_high_q = df["Arousal_mean"].quantile(1 - percentile_p)

    result: Dict[str, Any] = {}
    for name in image_names:
        rec = records.get(name)

        valence = float(rec["Valence_mean"])
        arousal = float(rec["Arousal_mean"])

        if valence <= v_low_q:
            valence_group = "low"
        elif valence >= v_high_q:
            valence_group = "high"
        else:
            valence_group = "mid"

        if arousal <= a_low_q:
            arousal_group = "low"
        elif arousal >= a_high_q:
            arousal_group = "high"
        else:
            arousal_group = "mid"

        result[name] = {
            "img": name,
            "Valence_mean": valence,
            "Arousal_mean": arousal,
            "valence_group": valence_group,
            "arousal_group": arousal_group,
        }

    return result


def main():
    oasis_df = load_oasis()
    image_selector(oasis_categories, img_per_category, percentile, banned_words, oasis_df=oasis_df)

    for cat, img_names in oasis_categories.items():
        print(f"Category: {cat}, Images: {len(img_names)}")

    # chechk total unique images selected
    selected_imgs = []
    for cat_imgs in oasis_categories.values():
        selected_imgs.extend(cat_imgs)
    selected_imgs = list(set(selected_imgs))

    expected_num = img_per_category * len(oasis_categories)
    assert len(selected_imgs) == expected_num  # may get duplicates, better to run again until it passes

    # bonus image that gets shown at the beggining
    bonus_img = sample_images_by_valence_arousal(
        oasis_df, mode="both", n=10, percentile_p=0.1, valence_group="mid", arousal_group="mid"
    )
    # Selects first image not in selected_imgs and not in banned words
    bw_lower = tuple(bw.lower() for bw in banned_words)  # tuple for startswith
    s = bonus_img["img"].astype("string")

    mask = ~s.isin(selected_imgs) & (~s.str.lower().str.startswith(bw_lower) if bw_lower else True)

    bonus_row = bonus_img[mask].head(1)  # single-row DataFrame
    # print(f"Bonus image selected: {bonus_row}")

    # defines order of images, starting with bonus image
    img_order = [bonus_row.iloc[0]["img"]] + np.random.permutation(selected_imgs).tolist()
    print(img_order)

    # give each selected image a valence/arousal group
    secondary_categories = categorize_selected_imgs(selected_imgs, oasis_df)
    # print(secondary_categories)
    # print(oasis_categories)
    sec_hv = [img for img, rec in secondary_categories.items() if rec["valence_group"] == "high"]
    sec_mv = [img for img, rec in secondary_categories.items() if rec["valence_group"] == "mid"]
    sec_lv = [img for img, rec in secondary_categories.items() if rec["valence_group"] == "low"]

    sec_ha = [img for img, rec in secondary_categories.items() if rec["arousal_group"] == "high"]
    sec_ma = [img for img, rec in secondary_categories.items() if rec["arousal_group"] == "mid"]
    sec_la = [img for img, rec in secondary_categories.items() if rec["arousal_group"] == "low"]

    oasis_sec_categories = {
        "high_valence": sec_hv,
        "mid_valence": sec_mv,
        "low_valence": sec_lv,
        "high_arousal": sec_ha,
        "mid_arousal": sec_ma,
        "low_arousal": sec_la,
    }
    all_info = {}
    all_info["oasis_categories"] = oasis_categories
    all_info["oasis_sec_categories"] = oasis_sec_categories
    all_info["img_w_cat"] = secondary_categories
    all_info["img"] = selected_imgs
    all_info["img_order"] = img_order
    out_path = Path(__file__).with_name("all_info.pkl")

    # save to file if "img" folder does not exist
    if not os.path.exists(out_img_folder):
        os.mkdir(out_img_folder)
        with out_path.open("wb") as f:
            pickle.dump(all_info, f, protocol=pickle.HIGHEST_PROTOCOL)
        # print(f"Saved all_info to {out_path}")
        for img_name in img_order:
            src_path = Path(oasis_csv_path).parent / "images" / f"{img_name}.jpg"
            dst_path = Path(out_img_folder) / f"{img_name}.jpg"
            if src_path.exists():
                img = PILImage.open(src_path)
                img.save(dst_path)
            else:
                print(f"Warning: source image not found: {src_path}")

    # for img_name, rec in secondary_categories.items():
    #     print(
    #         f"Image: {img_name}, Valence: {rec['Valence_mean']:.2f} ({rec['valence_group']}), Arousal: {rec['Arousal_mean']:.2f} ({rec['arousal_group']})"
    #     )


if __name__ == "__main__":
    main()
