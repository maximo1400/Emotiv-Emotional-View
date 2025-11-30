import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os, shutil
import pickle
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Patch
from itertools import zip_longest
from matplotlib.patches import Patch as _Patch

warnings.filterwarnings("ignore")

# fmt: off
EEG_CONFIG = {
    "frontal_pairs": [
        ("F3", "F4"),  # Primary frontal asymmetry
        ("AF3", "AF4"),  # Anterior frontal
        ("F7", "F8"),  # Lateral frontal
    ],
    "frontal_electrodes": ["F3", "F4", "AF3", "AF4", "F7", "F8", "FC5", "FC6"],
    "parietal_pairs": [
        ("P7", "P8"),  # Parietal asymmetry
    ],
    "parietal_electrodes": ["P7", "P8"],
    "electrodes": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"],
    "frequency_bands": ["alpha", "betaL", "betaH", "theta", "gamma"],
    "pow_columns": [
        "AF3/theta", "AF3/alpha", "AF3/betaL", "AF3/betaH", "AF3/gamma",
        "F7/theta",  "F7/alpha",  "F7/betaL",  "F7/betaH",  "F7/gamma",
        "F3/theta",  "F3/alpha",  "F3/betaL",  "F3/betaH",  "F3/gamma",
        "FC5/theta", "FC5/alpha", "FC5/betaL", "FC5/betaH", "FC5/gamma",
        "T7/theta",  "T7/alpha",  "T7/betaL",  "T7/betaH",  "T7/gamma",
        "P7/theta",  "P7/alpha",  "P7/betaL",  "P7/betaH",  "P7/gamma",
        "O1/theta",  "O1/alpha",  "O1/betaL",  "O1/betaH",  "O1/gamma",
        "O2/theta",  "O2/alpha",  "O2/betaL",  "O2/betaH",  "O2/gamma",
        "P8/theta",  "P8/alpha",  "P8/betaL",  "P8/betaH",  "P8/gamma",
        "T8/theta",  "T8/alpha",  "T8/betaL",  "T8/betaH",  "T8/gamma",
        "FC6/theta", "FC6/alpha", "FC6/betaL", "FC6/betaH", "FC6/gamma",
        "F4/theta",  "F4/alpha",  "F4/betaL",  "F4/betaH",  "F4/gamma",
        "F8/theta",  "F8/alpha",  "F8/betaL",  "F8/betaH",  "F8/gamma",
        "AF4/theta", "AF4/alpha", "AF4/betaL", "AF4/betaH", "AF4/gamma"],
    "time_s": 10,  # Duration each image is shown in seconds plus rest periods
    "sampling_rate": 8,  # Hz
    "target_samples_per_image": 80,  # Target samples per image for balancing
}

img_info_path = "OASIS_database_2016/all_info.pkl"
input_dir = "sub_data"
output_dir = "sub_data/emotions"
# fmt: on


def load_eeg_data(filename: str) -> pd.DataFrame:
    """Load EEG data from CSV file"""
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df["img"] = df["img"].str.strip()
    # Remove the first unnamed column if it exists
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(df.columns[0], axis=1)
    df = df.drop("round", axis=1)
    df = df[~df["img"].eq("end")]
    return df


def load_image_info(pickle_path: str = img_info_path) -> pd.DataFrame:
    """Load image information from pickle file"""
    with open(pickle_path, "rb") as f:
        img_info = pickle.load(f)
    all_info = {}
    all_info["oasis_categories"] = img_info["oasis_categories"]
    all_info["oasis_sec_categories"] = img_info["oasis_sec_categories"]
    all_info["img_w_cat"] = img_info["img_w_cat"]
    # all_info["img"] = img_info["img"]
    # all_info["img_order"] = img_info["img_order"]

    return all_info


def clean_slice_df(df: pd.DataFrame, eeg_config: dict = EEG_CONFIG) -> pd.DataFrame:
    """Clean the DataFrame by removing rows with NaN or infinite values and balance image groups.

    Behaviour:
    - Target a fixed per-image sample count (from EEG_CONFIG).
    - For images with >= TARGET_SAMPLES use the last TARGET_SAMPLES rows.
    - For images with fewer than TARGET_SAMPLES keep whatever rows are available.
    - R2 is treated as a fixed-size segment (nominal_R2_len) whenever possible;
      when samples are insufficient, R1 is reduced first (I is preserved if possible).
    """
    time_s = eeg_config["time_s"]
    target_samples = eeg_config["target_samples_per_image"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    # compute per-image counts once
    # min_samples = df.groupby("img").size().min()
    # print(f"Actual min samples available: {min_samples}.")

    balanced = df.copy().reset_index(drop=False)  # keep original index in 'index' column for safe loc
    balanced["slice"] = pd.NA

    # nominal slice sizes based on target_samples and time proportions
    nominal_I = int(round(target_samples * 6.0 / time_s))  # preserve primary image period
    nominal_R1 = int(round(target_samples * 8.0 / time_s))  # end of R1 at 8s
    nominal_R2_len = max(0, target_samples - nominal_R1)  # R2 nominally covers last (10-8)=2s

    for img in balanced["img"].unique():
        rows = balanced[balanced["img"] == img]
        orig_indices = rows["index"].tolist()  # original df indices for these rows
        n = len(orig_indices)

        selected_idx = orig_indices.copy()

        # allocate slices within selected_idx
        I = nominal_I
        R2 = nominal_R2_len
        R1_len = target_samples - I - R2
        # indices relative to selected_idx
        I_idx = selected_idx[:I]
        R1_idx = selected_idx[I : I + R1_len]
        R2_idx = selected_idx[-R2:]

        # assign slices
        balanced.loc[balanced["index"].isin(I_idx), "slice"] = "I"
        balanced.loc[balanced["index"].isin(R1_idx), "slice"] = "R1"
        balanced.loc[balanced["index"].isin(R2_idx), "slice"] = "R2"

        # remove rows of the current image that did not receive a slice assignment (mostly around end of R1)
        mask_unassigned = (balanced["img"] == img) & balanced["slice"].isna()
        if mask_unassigned.any():
            balanced = balanced.loc[~mask_unassigned].reset_index(drop=True)

    # drop helper 'index' column and return contiguous frame
    balanced = balanced.drop(columns=["index"]).reset_index(drop=True)
    # balanced.to_csv(f"{output_dir}/cleaned_balanced_eeg_data.csv", index=False)
    return balanced


def calculate_asymetryies(
    df: pd.DataFrame,
    df_valanced: pd.DataFrame,
    eeg_config=EEG_CONFIG,
    eps: float = 1e-10,
) -> dict:
    """
    Compute EEG asymmetries for each frequency band using multiple methods.
    """
    results = {}
    results["methods"] = ["standard", "normalized", "ratio", "ratio_norm"]

    # Define electrode pairs for asymmetry calculations
    frontal_pairs = eeg_config["frontal_pairs"]
    parietal_pairs = eeg_config["parietal_pairs"]
    frequency_bands = eeg_config["frequency_bands"]

    # Single merged loop for all asymmetry calculations
    for band in frequency_bands:
        # Initialize lists for different calculation methods
        frontal_asym = []
        frontal_asym_norm = []

        frontal_asym_rational = []
        frontal_asym_ratio_norm = []

        parietal_asym = []
        parietal_asym_norm = []

        parietal_asym_rational = []
        parietal_asym_ratio_norm = []

        # Calculate asymmetries with all methods
        for pair_type, pairs in [
            ("frontal", frontal_pairs),
            ("parietal", parietal_pairs),
        ]:
            for left, right in pairs:
                left_col = f"{left}/{band}"
                right_col = f"{right}/{band}"

                # Calculate total power for normalization (more robust)
                left_total = 0
                right_total = 0

                for freq in frequency_bands:
                    left_total += df[f"{left}/{freq}"]
                    right_total += df[f"{right}/{freq}"]

                # Raw values with epsilon for numerical stability
                left_raw = df[left_col] + eps
                right_raw = df[right_col] + eps

                # Normalized values
                left_norm = df[left_col] / (left_total + eps)
                right_norm = df[right_col] / (right_total + eps)

                # Method 1: Standard log asymmetry
                asymmetry_log = np.log(right_raw) - np.log(left_raw)

                # Method 2: Normalized log asymmetry
                asymmetry_norm_log = np.log(right_norm + eps) - np.log(left_norm + eps)

                # Method 3: Raw ratio asymmetry
                ratio_raw = right_raw / left_raw

                # Method 4: Normalized ratio asymmetry
                ratio_norm = right_norm / (left_norm + eps)

                if pair_type == "frontal":
                    frontal_asym.append(asymmetry_log)
                    frontal_asym_norm.append(asymmetry_norm_log)
                    frontal_asym_rational.append(ratio_raw)
                    frontal_asym_ratio_norm.append(ratio_norm)
                else:  # Parietal
                    parietal_asym.append(asymmetry_log)
                    parietal_asym_norm.append(asymmetry_norm_log)
                    parietal_asym_rational.append(ratio_raw)
                    parietal_asym_ratio_norm.append(ratio_norm)

        # Store averaged results for each method
        results[f"frontal_asym_{band}"] = np.mean(frontal_asym, axis=0)
        results[f"frontal_asym_norm_{band}"] = np.mean(frontal_asym_norm, axis=0)

        results[f"frontal_asym_ratio_{band}"] = np.mean(frontal_asym_rational, axis=0)
        results[f"frontal_asym_ratio_norm_{band}"] = np.mean(frontal_asym_ratio_norm, axis=0)

        results[f"parietal_asym_{band}"] = np.mean(parietal_asym, axis=0)
        results[f"parietal_asym_norm_{band}"] = np.mean(parietal_asym_norm, axis=0)

        results[f"parietal_asym_ratio_{band}"] = np.mean(parietal_asym_rational, axis=0)
        results[f"parietal_asym_ratio_norm_{band}"] = np.mean(parietal_asym_ratio_norm, axis=0)

    return results


def calculate_valence(df: pd.DataFrame, asymmetries: dict, eeg_config=EEG_CONFIG, eps: float = 1e-10) -> dict:
    """
    Calculate valence and return results tied to the image IDs in df.
    Each result is provided both as a per-row pd.Series (index = df['img'])
    and as an aggregated per-image series (mean across rows for each img).
    """
    results = {}

    # build raw method arrays (aligned with df rows)
    valence_methods = []

    # Valence from different alpha asymmetry methods
    valence_methods.append(-asymmetries["frontal_asym_alpha"])  # Standard log
    valence_methods.append(-asymmetries["frontal_asym_norm_alpha"])  # Normalized log

    # For ratio methods, convert to log scale for consistency
    ratio_raw = asymmetries["frontal_asym_ratio_alpha"]
    ratio_norm = asymmetries["frontal_asym_ratio_norm_alpha"]
    valence_methods.append(-np.log(ratio_raw + eps))  # Ratio as log
    valence_methods.append(-np.log(ratio_norm + eps))

    # Composite score (weighted) or fallback mean
    if len(valence_methods) == 4:
        weights = [0.3, 0.3, 0.2, 0.2]
        composite_arr = np.average(valence_methods, weights=weights, axis=0)
    else:
        composite_arr = np.mean(valence_methods, axis=0)

    # Create pandas Series tied to img for each method + composite
    img_index = df["img"].reset_index(drop=True)
    # store composite
    composite_series = pd.Series(composite_arr, index=img_index)
    results["valence"] = composite_series
    results["valence_img_mean"] = composite_series.groupby(level=0).mean()

    # Store individual methods labeled by asymmetry method names (from asymmetries["methods"])
    method_names = asymmetries["methods"]
    for i, method in enumerate(method_names[: len(valence_methods)]):
        arr = valence_methods[i]
        ser = pd.Series(arr, index=img_index)
        results[f"valence_{method}"] = ser
        # aggregated per image (mean)
        results[f"valence_{method}_img_mean"] = ser.groupby(level=0).mean()

    return results


def calculate_arousal(df: pd.DataFrame, asymmetries: dict, eeg_config=EEG_CONFIG, eps: float = 1e-10) -> dict:
    results = {}

    # Enhanced arousal calculation
    frontal_electrodes = eeg_config["frontal_electrodes"]

    # Multiple arousal measures (collect as numpy arrays for consistency)
    beta_low_activities = [df[f"{electrode}/betaL"].to_numpy() for electrode in frontal_electrodes]
    beta_high_activities = [df[f"{electrode}/betaH"].to_numpy() for electrode in frontal_electrodes]
    beta_combined_activities = [low + high for low, high in zip(beta_low_activities, beta_high_activities)]

    # Average across electrodes (axis=0 -> per-timepoint mean across electrodes)
    arousal_beta_low_arr = np.mean(beta_low_activities, axis=0)
    arousal_beta_high_arr = np.mean(beta_high_activities, axis=0)
    arousal_beta_combined_arr = np.mean(beta_combined_activities, axis=0)

    # Primary arousal measure (combined beta for robustness)
    arousal_arr = arousal_beta_combined_arr

    # Build pandas Series tied to img (like calculate_valence)
    img_index = df["img"].reset_index(drop=True)

    ser_low = pd.Series(arousal_beta_low_arr, index=img_index)
    ser_high = pd.Series(arousal_beta_high_arr, index=img_index)
    ser_combined = pd.Series(arousal_beta_combined_arr, index=img_index)
    ser_primary = pd.Series(arousal_arr, index=img_index)

    # Store per-row series
    results["arousal_beta_low"] = ser_low
    results["arousal_beta_high"] = ser_high
    results["arousal_beta_combined"] = ser_combined
    results["arousal"] = ser_primary

    # Store aggregated per-image means
    results["arousal_beta_low_img_mean"] = ser_low.groupby(level=0).mean()
    results["arousal_beta_high_img_mean"] = ser_high.groupby(level=0).mean()
    results["arousal_beta_combined_img_mean"] = ser_combined.groupby(level=0).mean()
    results["arousal_img_mean"] = ser_primary.groupby(level=0).mean()

    return results


def calculate_dominance(df: pd.DataFrame, asymmetries: dict, eeg_config=EEG_CONFIG, eps: float = 1e-10) -> dict:
    results = {}

    # Calculate final metrics with multiple approaches
    dominance_methods = []

    # Dominance from parietal asymmetries
    dominance_methods.append(asymmetries["parietal_asym_alpha"])
    dominance_methods.append(asymmetries["parietal_asym_norm_alpha"])
    dominance_methods.append(np.log(asymmetries["parietal_asym_ratio_alpha"] + eps))
    dominance_methods.append(np.log(asymmetries["parietal_asym_ratio_norm_alpha"] + eps))

    # Dominance calculation (per-row numpy array)
    dominance_arr = np.mean(dominance_methods, axis=0)

    # Build pandas Series tied to img (like calculate_valence / calculate_arousal)
    img_index = df["img"].reset_index(drop=True)
    ser_primary = pd.Series(dominance_arr, index=img_index)

    # Store per-row series and aggregated per-image mean
    results["dominance"] = ser_primary
    results["dominance_img_mean"] = ser_primary.groupby(level=0).mean()

    # Store individual methods for comparison (per-row + per-image mean)
    method_names = asymmetries.get("methods", [])
    for i in range(min(len(method_names), len(dominance_methods))):
        arr = dominance_methods[i]
        ser = pd.Series(arr, index=img_index)
        key = f"dominance_{method_names[i]}"
        results[key] = ser
        results[f"{key}_img_mean"] = ser.groupby(level=0).mean()

    return results


def plot_valence_arousal(
    vda_results,
    save_plot=True,
    dominance_is_size=True,
    bin_size=1,
    output_dir=output_dir,
):
    """
    Create a scatter plot of valence vs arousal with quadrant analysis
    """
    if "valence" not in vda_results or "arousal" not in vda_results or "dominance" not in vda_results:
        print("Error: Valence, arousal, or dominance data not available for plotting")
        return

    valence = vda_results["valence"]
    arousal = vda_results["arousal"]
    dominance = vda_results["dominance"]

    n_points = len(valence)
    n_bins = n_points // bin_size

    valence_binned = []
    arousal_binned = []
    dominance_binned = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size

        valence_binned.append(np.mean(valence[start_idx:end_idx]))
        arousal_binned.append(np.mean(arousal[start_idx:end_idx]))
        dominance_binned.append(np.mean(dominance[start_idx:end_idx]))

    # Convert to numpy arrays
    valence = np.array(valence_binned)
    arousal = np.array(arousal_binned)
    dominance = np.array(dominance_binned)

    print(f"Reduced from {n_points} to {len(valence)} points")

    sizes = 50
    if dominance_is_size:
        dom_min, dom_max = np.min(dominance), np.max(dominance)
        if dom_max != dom_min:  # Avoid division by zero
            sizes = 1 + 100 * (dominance - dom_min) / (dom_max - dom_min)

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    plt.scatter(
        valence,
        arousal,
        alpha=0.7,
        s=sizes,
        c="blue",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add quadrant lines
    plt.axhline(
        y=np.mean(arousal),
        color="red",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label="Data mean",
    )
    plt.axvline(x=np.mean(valence), color="red", linestyle="--", alpha=0.6, linewidth=1)

    # Add quadrant lines at (0,0)
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.7, linewidth=2, label="Zero line")
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.7, linewidth=2)

    # Force (0,0) as the visual center
    # TODO : Use a more method with consistent axis limits
    x_absmax = max(abs(np.min(valence)), abs(np.max(valence)), 0.1) * 1.1
    y_absmax = max(abs(np.min(arousal)), abs(np.max(arousal)), 0.1) * 1.1
    axis_limit = max(x_absmax, y_absmax)
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)

    # Get current axis limits
    x_range = plt.xlim()
    y_range = plt.ylim()

    plt.text(
        x_range[1] * 0.8,
        y_range[1] * 0.9,
        "High Arousal\nPositive Valence\n(Happy/Excited)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[1] * 0.9,
        "High Arousal\nNegative Valence\n(Angry/Stressed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    plt.text(
        x_range[1] * 0.8,
        y_range[0] * 0.8,
        "Low Arousal\nPositive Valence\n(Calm/Relaxed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[0] * 0.8,
        "Low Arousal\nNegative Valence\n(Sad/Depressed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    # Customize the plot
    plt.xlabel("Valence (Negative ← → Positive)", fontsize=12, fontweight="bold")
    plt.ylabel("Arousal (Low ← → High)", fontsize=12, fontweight="bold")
    plt.title(
        "EEG-based Emotional State: Valence vs Arousal",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Data Points: {len(valence)}\n"
    stats_text += f"Valence: μ={np.mean(valence):.3f}, σ={np.std(valence):.3f}\n"
    stats_text += f"Arousal: μ={np.mean(arousal):.3f}, σ={np.std(arousal):.3f}\n"
    stats_text += f"Dominance: μ={np.mean(dominance):.3f}, σ={np.std(dominance):.3f}\n"

    # Add quadrant counts
    q1 = np.sum((valence >= 0) & (arousal >= 0))  # Happy
    q2 = np.sum((valence < 0) & (arousal >= 0))  # Angry
    q3 = np.sum((valence < 0) & (arousal < 0))  # Sad
    q4 = np.sum((valence >= 0) & (arousal < 0))  # Calm

    stats_text += f"Quadrants: Happy={q1}, Angry={q2}, Sad={q3}, Calm={q4}"

    plt.text(
        0.02,
        0.9,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], color="black", linestyle="-", label="Zero line"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Data mean"),
    ]

    if dominance_is_size:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="blue",
                markersize=8,
                label="Size ∝ Dominance",
            )
        )

    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.85))
    # plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85))

    plt.tight_layout()

    if save_plot:
        plt.savefig(f"{output_dir}/valence_arousal_plot.png", dpi=300, bbox_inches="tight")
        print("Plot saved as 'valence_arousal_plot.png'")

    plt.show()


def plot_time_series(vda_results, save_plot=True, output_dir=output_dir):
    """
    Create time series plots for valence, arousal, and dominance
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    metrics = ["valence", "arousal", "dominance"]
    colors = ["blue", "red", "green"]
    labels = [
        "Valence (Negative ← → Positive)",
        "Arousal (Low ← → High)",
        "Dominance (Submissive ← → Dominant)",
    ]

    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        if metric in vda_results:
            data = vda_results[metric]
            axes[i].plot(data, color=color, linewidth=1.5, alpha=0.8)
            axes[i].axhline(y=np.mean(data), color="black", linestyle="--", alpha=0.5)
            axes[i].set_ylabel(label, fontweight="bold")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"{metric.capitalize()} Over Time", fontweight="bold")

            # Add statistics
            stats_text = f"μ={np.mean(data):.3f}, σ={np.std(data):.3f}"
            axes[i].text(
                0.02,
                0.95,
                stats_text,
                transform=axes[i].transAxes,
                verticalalignment="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    axes[-1].set_xlabel("Time Points", fontweight="bold")
    plt.suptitle("EEG Emotional Metrics Time Series", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_plot:
        plt.savefig(
            f"{output_dir}/emotional_metrics_timeseries.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Time series plot saved as 'emotional_metrics_timeseries.png'")

    plt.show()


def plot_valence_arousal_methods(
    vda_results,
    save_plot=True,
    dominance_is_size=True,
    bin_size=1,
    output_dir=output_dir,
):
    """
    Create a scatter plot showing all valence and arousal calculation methods with different colors
    """
    # Define the methods we want to plot
    # valence_methods = ["valence_standard", "valence_normalized", "valence_ratio", "valence_ratio_norm"]
    valence_methods = ["valence_normalized", "valence_ratio_norm", "valence"]
    # arousal_methods = ["arousal_beta_low", "arousal_beta_high", "arousal_beta_combined"]
    arousal_methods = ["arousal_beta_low", "arousal"]

    # Check if methods exist in results
    available_valence = [method for method in valence_methods if method in vda_results]
    available_arousal = [method for method in arousal_methods if method in vda_results]

    # Define colors for different combinations
    # colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    # colors = list(plt.cm.tab20.colors)
    colors = list(plt.cm.Paired.colors)
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]

    # Create the plot
    plt.figure(figsize=(14, 10))

    dominance = vda_results["dominance"]

    # Apply binning to dominance
    n_points = len(dominance)
    n_bins = n_points // bin_size
    dominance_binned = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size
        dominance_binned.append(np.mean(dominance[start_idx:end_idx]))

    dominance = np.array(dominance_binned)

    # Calculate sizes for dominance
    sizes = 50
    if dominance_is_size:
        dom_min, dom_max = np.min(dominance), np.max(dominance)
        if dom_max != dom_min:
            sizes = 10 + 100 * (dominance - dom_min) / (dom_max - dom_min)

    # Plot all combinations
    plot_idx = 0
    all_valence = []
    all_arousal = []
    custom_legend_elements = []

    for v_method in available_valence:
        for a_method in available_arousal:
            if plot_idx >= len(colors):
                break

            # Get data
            valence_data = vda_results[v_method]
            arousal_data = vda_results[a_method]

            # Apply binning
            valence_binned = []
            arousal_binned = []

            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = start_idx + bin_size
                valence_binned.append(np.mean(valence_data[start_idx:end_idx]))
                arousal_binned.append(np.mean(arousal_data[start_idx:end_idx]))

            valence_binned = np.array(valence_binned)
            arousal_binned = np.array(arousal_binned)

            # Store for axis limits calculation
            all_valence.extend(valence_binned)
            all_arousal.extend(arousal_binned)

            # Create scatter plot
            color = colors[plot_idx]
            marker = markers[plot_idx % len(markers)]

            # Create cleaner label
            v_label = v_method.replace("valence_", "").replace("_", " ").title()
            a_label = a_method.replace("arousal_", "").replace("_", " ").title()
            label = f"{v_label} + {a_label}"

            plt.scatter(
                valence_binned,
                arousal_binned,
                alpha=0.6,
                s=sizes,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidth=0.3,
                # label=f"{v_method.replace('valence_', 'V:')} + {a_method.replace('arousal_', 'A:')}",
                # label=label,
            )
            custom_legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    label=label,
                )
            )
            plot_idx += 1

    print(f"Plotted {plot_idx} method combinations")
    print(f"Reduced from {n_points} to {len(dominance)} points per method")

    # Convert to numpy arrays for axis calculations
    all_valence = np.array(all_valence)
    all_arousal = np.array(all_arousal)

    # Add reference lines
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=2, label="Zero line")
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.8, linewidth=2)
    plt.axhline(
        y=np.mean(all_arousal),
        color="red",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label="Overall mean",
    )
    plt.axvline(x=np.mean(all_valence), color="red", linestyle="--", alpha=0.6, linewidth=1)

    # Force (0,0) as the visual center
    x_absmax = max(abs(np.min(all_valence)), abs(np.max(all_valence)), 0.1) * 1.1
    y_absmax = max(abs(np.min(all_arousal)), abs(np.max(all_arousal)), 0.1) * 1.1
    axis_limit = max(x_absmax, y_absmax)
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)

    # Get current axis limits
    x_range = plt.xlim()
    y_range = plt.ylim()

    # Add quadrant labels
    plt.text(
        x_range[1] * 0.8,
        y_range[1] * 0.9,
        "High Arousal\nPositive Valence\n(Happy/Excited)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[1] * 0.9,
        "High Arousal\nNegative Valence\n(Angry/Stressed)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    plt.text(
        x_range[1] * 0.8,
        y_range[0] * 0.8,
        "Low Arousal\nPositive Valence\n(Calm/Relaxed)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[0] * 0.8,
        "Low Arousal\nNegative Valence\n(Sad/Depressed)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    # Customize the plot
    plt.xlabel("Valence (Negative ← → Positive)", fontsize=12, fontweight="bold")
    plt.ylabel("Arousal (Low ← → High)", fontsize=12, fontweight="bold")
    plt.title(
        "EEG Emotional State: All Calculation Methods Comparison",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Methods plotted: {plot_idx}\n"
    stats_text += f"Points per method: {len(dominance)}\n"
    stats_text += f"Valence range: [{np.min(all_valence):.3f}, {np.max(all_valence):.3f}]\n"
    stats_text += f"Arousal range: [{np.min(all_arousal):.3f}, {np.max(all_arousal):.3f}]\n"
    stats_text += f"Dominance: μ={np.mean(dominance):.3f}, σ={np.std(dominance):.3f}"

    plt.text(
        0.02,
        0.9,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    # Add reference lines to legend
    custom_legend_elements.append(plt.Line2D([0], [0], color="black", linestyle="-", label="Zero line"))
    custom_legend_elements.append(plt.Line2D([0], [0], color="red", linestyle="--", label="Mean line"))

    if dominance_is_size:
        custom_legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=8,
                label="Size ∝ Dominance",
            )
        )

    # Create legend in the plot area
    plt.legend(
        handles=custom_legend_elements,
        loc="upper right",
        bbox_to_anchor=(1, 0.9),
        fontsize=8,
        framealpha=0.9,
    )

    plt.tight_layout()

    if save_plot:
        plt.savefig(
            f"{output_dir}/valence_arousal_methods_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'valence_arousal_methods_comparison.png'")

    plt.show()

    # Print method summary
    print("\nMethod combinations plotted:")
    plot_idx = 0
    for v_method in available_valence:
        for a_method in available_arousal:
            if plot_idx >= len(colors):
                break
            print(f"{plot_idx + 1}. {v_method} + {a_method} (Color: {colors[plot_idx]})")
            plot_idx += 1


def drop_first_image(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the very first image (warm-up artifacts)."""
    first_img = df["img"].iloc[0]
    return df[df["img"] != first_img].reset_index(drop=True)


def compute_prev_R2_baseline(df: pd.DataFrame, eeg_config=EEG_CONFIG) -> pd.DataFrame:
    """
    Baseline for each image = median of previous image's R2.
    Fallback = session median if previous R2 is missing (or for the first image).
    Returns a DataFrame indexed by img with baseline vectors for pow columns.
    Assumes df['slice'] already has R2 labels and images are unique.
    """
    bands = eeg_config["pow_columns"]

    # Median per image over R2
    r2_per_img = df[df["slice"] == "R2"].groupby("img", sort=False)[bands].mean()

    # Presentation order from file
    order = df.drop_duplicates("img", keep="first")["img"].tolist()
    session_med = df[bands].mean()

    aligned = {}
    for i, img in enumerate(order):
        prev_img = order[i - 1] if i > 0 else None
        if prev_img is not None and prev_img in r2_per_img.index:
            aligned[img] = r2_per_img.loc[prev_img]
        else:
            aligned[img] = session_med

    base = pd.DataFrame(aligned).T
    base.index.name = "img"
    return base  # index=img, columns=bands


def apply_baseline_percent_change(
    df: pd.DataFrame,
    base_prev_r2: pd.DataFrame,
    eps: float = 1e-10,
    eeg_config=EEG_CONFIG,
) -> pd.DataFrame:
    """
    Percent change relative to previous R2: (value - base) / (|base| + eps).
    Use this if you prefer relative normalization.
    """
    df = df.copy()
    bands = eeg_config["pow_columns"]

    def row_op(df_local):
        baseline = base_prev_r2.loc[df_local["img"], bands]
        return (df_local[bands] - baseline) / (baseline.abs() + eps)

    df[bands] = df.apply(row_op, axis=1)
    return df


def plot_time_serie_with_images(
    vda: dict,
    df: pd.DataFrame,
    img_seq: list,
    img_category_map: dict,
    calc_method: str = "arousal_norm",
    emotion_type: str = "arousal",
    save_plot: bool = True,
    output_dir: str = output_dir,
    figsize=(14, 6),
    connect_segments: bool = False,
    line_color: str = "tab:blue",
):
    """
    Plot the valence/arousal time series with image labels and background coloring for
    images in high_val_list (green) and low_val_list (red).

    If img_seq is provided (non-empty), reorder the rows so image blocks follow img_seq,
    preserving within-image time order. Any images present in df but not in img_seq are
    appended after the seq in their original order.
    """
    high_list = set(img_category_map.get(f"high_{emotion_type}", []) or [])
    low_list = set(img_category_map.get(f"low_{emotion_type}", []) or [])

    # Original imgs and series
    imgs_orig = df["img"].reset_index(drop=True).astype(str).tolist()
    serie_raw = vda.get(calc_method)

    # Build ordered index using img_seq if provided
    ordered_idxs = []
    if img_seq:
        seen = set()
        # Add indices for images in img_seq in that order
        for img in img_seq:
            matches = df.index[df["img"].astype(str) == str(img)].tolist()
            for idx in matches:
                if idx not in seen:
                    ordered_idxs.append(idx)
                    seen.add(idx)
        # Append any remaining indices (images in df not in img_seq), preserving original order
        for idx in df.index.tolist():
            if idx not in seen:
                ordered_idxs.append(idx)
                seen.add(idx)
    else:
        ordered_idxs = df.index.tolist()

    if not ordered_idxs:
        print("No data to plot.")
        return

    # Reorder df
    df_ord = df.loc[ordered_idxs].reset_index(drop=True)

    # Prepare y (series) aligned to df_ord
    serie_s = pd.Series(serie_raw).reset_index(drop=True)

    # reorder per-row series
    y_series = serie_s.iloc[ordered_idxs].reset_index(drop=True)

    # Final numpy arrays for plotting
    y = y_series.reset_index(drop=True).to_numpy()
    n = len(y)
    x = np.arange(n)
    imgs = df_ord["img"].reset_index(drop=True).astype(str).tolist()

    # find contiguous segments of the same image
    segments = []
    start = 0
    cur_img = imgs[0] if imgs else ""
    for i in range(1, n):
        if imgs[i] != cur_img:
            segments.append((cur_img, start, i))  # end is exclusive
            start = i
            cur_img = imgs[i]
    if n > 0:
        segments.append((cur_img, start, n))

    fig, ax = plt.subplots(figsize=figsize)

    # Plot either as one continuous line or as separate segment lines
    if connect_segments:
        ax.plot(x, y, color=line_color, linewidth=1.5, alpha=0.9)
    else:
        # plot each segment separately so there is a gap between segments
        for img_id, s, e in segments:
            if e - s <= 0:
                continue
            seg_x = np.arange(s, e)
            seg_y = y[s:e]
            if len(seg_x) == 1:
                ax.plot(seg_x, seg_y, marker="o", color=line_color, markersize=4, alpha=0.9, linestyle="None")
            else:
                ax.plot(seg_x, seg_y, color=line_color, linewidth=1.5, alpha=0.9)

    # Plot mean line
    # ax.axhline(y=np.nanmean(y) if n > 0 else 0.0, color="black", linestyle="--", alpha=0.6, linewidth=1)

    # shading and labels
    high_color = "lightgreen"
    low_color = "lightcoral"

    # compute y placement for labels (below the series)
    finite_y = y[np.isfinite(y)] if n > 0 else np.array([0.0])
    if finite_y.size == 0:
        y_min, y_max = 0.0, 0.0
    else:
        y_min, y_max = np.min(finite_y), np.max(finite_y)
    y_range = max(1e-6, y_max - y_min)
    label_y = y_min - 0.06 * y_range  # put labels slightly below the minimum
    # expand y limits to ensure labels are visible
    ylim_min = y_min - 0.12 * y_range
    ylim_max = y_max + 0.06 * y_range
    ax.set_ylim(ylim_min, ylim_max)

    for img_id, s, e in segments:
        # choose color
        col = None
        if img_id in high_list:
            col = high_color
        elif img_id in low_list:
            col = low_color

        if col:
            ax.axvspan(s - 0.5, e - 0.5, facecolor=col, alpha=0.25, edgecolor=None)

        # label in the center of the segment: show img id and optionally category
        label = str(img_id)
        if img_category_map and isinstance(img_category_map, dict):
            cat = img_category_map.get(img_id)
            if cat is not None:
                label = f"{img_id}\n{cat}"

        seg_len = e - s
        if seg_len >= 1:
            ax.text((s + e - 1) / 2.0, label_y, label, ha="center", va="top", fontsize=7, rotation=0)

    # Decorations
    ax.set_xlabel("Time Points", fontweight="bold")
    if emotion_type == "arousal":
        ax.set_ylabel("Arousal (Low ← → High)", fontweight="bold")
        ax.set_title(f"Arousal ({calc_method}) Over Time (Images annotated & highlighted)", fontweight="bold")
    else:
        ax.set_ylabel("Valence (Negative ← → Positive)", fontweight="bold")
        ax.set_title(f"Valence ({calc_method}) Over Time (Images annotated & highlighted)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # build legend for shaded regions if used
    legend_handles = []
    if high_list:
        legend_handles.append(Patch(facecolor=high_color, alpha=0.25, label=f"High {emotion_type} images"))
    if low_list:
        legend_handles.append(Patch(facecolor=low_color, alpha=0.25, label=f"Low {emotion_type} images"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    plt.tight_layout()
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/{calc_method}_time_with_images.png", dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_dir}/{calc_method}_time_with_images.png")

    plt.show()
    return


def filter_and_alternate_images(
    df: pd.DataFrame,
    vda: dict,
    img_cat_map: dict,
    seq: list,
    emot_type: str = "valence",
    start_with: str = "high",
):
    """
    Filter df and vda to only rows whose img is in high_val_list or low_val_list,
    then reorder them so image blocks alternate between high and low lists
    (preserving the original within-image row order). Returns (df_filtered, vda_filtered, seq)
    where seq is the final image sequence used.
    """
    high_val_list = img_cat_map[f"high_{emot_type}"]
    low_val_list = img_cat_map[f"low_{emot_type}"]

    # Ensure lists are sets for quick membership
    high_set = set(high_val_list or [])
    low_set = set(low_val_list or [])
    allowed = high_set | low_set

    # Original indices (time order) and unique images in their first-appearance order
    orig_idx = df.index.to_list()
    unique_imgs = df.drop_duplicates("img", keep="first")["img"].astype(str).tolist()

    # Keep only those images that are present in df and in the allowed sets
    high_present = [img for img in unique_imgs if img in high_set]
    low_present = [img for img in unique_imgs if img in low_set]

    if not high_present and not low_present:
        # Nothing to do: return copies
        return df.copy(), {k: (v.copy() if isinstance(v, pd.Series) else v) for k, v in vda.items()}, []

    # Interleave lists preserving internal order
    if seq == []:
        if start_with == "low":
            a, b = low_present, high_present
        else:
            a, b = high_present, low_present

        for x, y in zip_longest(a, b):
            if x is not None:
                seq.append(x)
            if y is not None:
                seq.append(y)

        # Remove potential duplicates while preserving order (rare)
        seen = set()
        seq_ordered = []
        for s in seq:
            if s not in seen:
                seen.add(s)
                seq_ordered.append(s)
        seq = seq_ordered

    # Build ordered list of original row indices according to seq (preserve within-image order)
    ordered_idxs = []
    for img in seq:
        rows = df.index[df["img"].astype(str) == str(img)].tolist()
        ordered_idxs.extend(rows)

    if not ordered_idxs:
        return df.copy(), {k: (v.copy() if isinstance(v, pd.Series) else v) for k, v in vda.items()}, seq

    # Create reordered df
    df_new = df.loc[ordered_idxs].reset_index(drop=True)

    # Reorder or expand vda entries to match df_new
    vda_new = {}
    orig_n = len(df)
    for key, val in vda.items():
        try:
            # pandas Series handling
            if isinstance(val, pd.Series):
                val_pos = val.reset_index(drop=True)
                if len(val_pos) == orig_n:
                    arr = val_pos.iloc[ordered_idxs].reset_index(drop=True)
                    vda_new[key] = arr
                    continue
                else:
                    # per-image Series (index=img)
                    imgs_in_new = df_new["img"].astype(str).tolist()
                    mapped = [val.get(i, np.nan) for i in imgs_in_new]
                    vda_new[key] = pd.Series(mapped)
                    continue
            # numpy / list per-row
            arr_np = np.asarray(val)
            if arr_np.size == orig_n:
                vda_new[key] = pd.Series(arr_np[ordered_idxs]).reset_index(drop=True)
            else:
                # attempt per-image mapping using position-aligned assumption (fallback)
                imgs_in_new = df_new["img"].astype(str).tolist()
                try:
                    # val might be dict-like
                    mapped = [val[i] if i in val else np.nan for i in imgs_in_new]
                except Exception:
                    mapped = [np.nan] * len(imgs_in_new)
                vda_new[key] = pd.Series(mapped)
        except Exception:
            # Safe fallback: drop or fill with NaN series of correct length
            vda_new[key] = pd.Series([np.nan] * len(df_new))

    return df_new, vda_new, seq


def init_shared_plot(
    img_seq: list,
    img_category_map: dict,
    calc_method: str = "arousal",
    emotion_type: str = "arousal",
    figsize=(18, 6),
    output_dir: str = output_dir,
    cmap: str = "tab20",
    show_images: bool = False,
    mode: str = "both",  # "mean", "individual", or "both"
):
    """
    Initialize a shared figure/axis for plotting multiple people's time series.
    Returns a dict `shared` holding fig, ax, color cycle and config.

    mode:
        - "mean":      plot only group-mean line
        - "individual": plot only individual lines
        - "both":      plot individual lines + group-mean line
    """
    fig, ax = plt.subplots(figsize=figsize)
    cmap_obj = plt.get_cmap(cmap)
    colors = list(cmap_obj.colors) if hasattr(cmap_obj, "colors") else [cmap_obj(i) for i in range(20)]
    high_list = set(img_category_map.get(f"high_{emotion_type}", []) or [])
    low_list = set(img_category_map.get(f"low_{emotion_type}", []) or [])
    shared = {
        "fig": fig,
        "ax": ax,
        "colors": colors,
        "color_idx": 0,
        "calc_method": calc_method,
        "emotion_type": emotion_type,
        "output_dir": output_dir,
        "show_images": show_images,
        "legend_handles": [],
        "labels": [],
        "img_order_labels": [],
        "img_seq": img_seq,
        "img_category_map": img_category_map,
        "high_list": high_list,
        "low_list": low_list,
        "all_y_values": [],
        "mode": mode,
    }
    # axis labels / title left to finalize_shared_plot, but set basic labels
    if emotion_type == "arousal":
        ax.set_ylabel("Arousal (Low ← → High)", fontweight="bold")
        ax.set_title(f"Combined Arousal ({calc_method}) Across People", fontweight="bold")
    else:
        ax.set_ylabel("Valence (Negative ← → Positive)", fontweight="bold")
        ax.set_title(f"Combined Valence ({calc_method}) Across People", fontweight="bold")
    ax.grid(True, alpha=0.3)
    return shared


def add_person_to_shared_plot(
    person_name: str,
    df: pd.DataFrame,
    vda_person: dict,
    shared: dict,
    connect_segments: bool = False,
    line_style: str = "-",
    linewidth: float = 1.5,
    alpha: float = 0.9,
    center_y_axis: bool = True,
):
    """
    Add one person's time series to a shared plot.
    df: dataframe for that person (must contain 'img' column)
    vda_person: dict of series/arrays for that person (must contain calc_method used in shared)
    """
    ax = shared["ax"]
    calc_method = shared["calc_method"]
    mode = shared.get("mode", "both")

    serie_raw = vda_person.get(calc_method)
    if serie_raw is None:
        print(f"Warning: {calc_method} not found for {person_name}; skipping.")
        return

    # Build ordered indices to make same ordering as plot_time_serie_with_images when img_seq supplied
    ordered_idxs = []
    img_seq = shared["img_seq"]
    if img_seq:
        seen = set()
        for img in img_seq:
            matches = df.index[df["img"].astype(str) == str(img)].tolist()
            for idx in matches:
                if idx not in seen:
                    ordered_idxs.append(idx)
                    seen.add(idx)
        for idx in df.index.tolist():
            if idx not in seen:
                ordered_idxs.append(idx)
                seen.add(idx)
    else:
        ordered_idxs = df.index.tolist()

    if not ordered_idxs:
        print(f"No data for {person_name}")
        return

    # Align y values to ordered_idxs
    try:
        if isinstance(serie_raw, pd.Series) and len(serie_raw) == len(df):
            y_series = serie_raw.reset_index(drop=True).iloc[ordered_idxs].reset_index(drop=True)
        else:
            # attempt to coerce to numpy and pick by ordered_idxs; if shorter assume per-image mapping
            arr = np.asarray(serie_raw)
            if arr.size == len(df):
                y_series = pd.Series(arr[ordered_idxs]).reset_index(drop=True)
            else:
                # fallback: map per-image (value per image) to rows in df via img column
                img_vals = []
                ser_map = serie_raw if isinstance(serie_raw, (dict, pd.Series)) else {}
                imgs_in_new = df.loc[ordered_idxs, "img"].astype(str).tolist()
                for img in imgs_in_new:
                    img_vals.append(ser_map.get(img, np.nan))
                y_series = pd.Series(img_vals)
    except Exception:
        # safest fallback: NaNs
        y_series = pd.Series([np.nan] * len(ordered_idxs))

    y = y_series.to_numpy()
    if center_y_axis:
        y = y - np.nanmean(y)
    shared["all_y_values"].append(y)
    n = len(y)
    x = np.arange(n)
    imgs = df.loc[ordered_idxs, "img"].reset_index(drop=True).astype(str).tolist()

    # choose color
    colors = shared["colors"]
    ci = shared["color_idx"] % len(colors)
    color = colors[ci]
    shared["color_idx"] += 1

    line = None
    # Plot as continuous or segmented (to reflect image blocks)
    if connect_segments and mode in ("individual", "both"):
        (line,) = ax.plot(x, y, color=color, linestyle=line_style, linewidth=linewidth, alpha=alpha, label=person_name)
    elif mode in ("individual", "both"):
        # find contiguous segments of same image to create visible gaps
        segments = []
        start = 0
        cur_img = imgs[0] if imgs else ""
        for i in range(1, n):
            if imgs[i] != cur_img:
                segments.append((start, i))
                start = i
                cur_img = imgs[i]
        if n > 0:
            segments.append((start, n))
        # plot segments separately
        line = None
        for s, e in segments:
            seg_x = np.arange(s, e)
            seg_y = y[s:e]
            if len(seg_x) == 1:
                (lobj,) = ax.plot(seg_x, seg_y, marker="o", color=color, markersize=4, linestyle="None", alpha=alpha)
                if line is None:
                    line = lobj
            else:
                (lobj,) = ax.plot(seg_x, seg_y, color=color, linestyle=line_style, linewidth=linewidth, alpha=alpha)
            if line is None:
                line = lobj

    # store handle and label for legend
    if line is not None:
        shared["legend_handles"].append(line)
        shared["labels"].append(person_name)

    # get ordered image labels if not yet defined
    img_labels = shared.get("img_order_labels", [])
    if not img_labels:
        img_labels = get_img_order_labels(df, img_seq)
        shared["img_order_labels"] = img_labels

    return


def get_img_order_labels(
    df: pd.DataFrame,
    img_seq: list,
) -> list:
    """
    Get ordered image labels consistent with img_seq and df.
    """
    ordered_idxs = []
    if img_seq:
        seen = set()
        for img in img_seq:
            matches = df.index[df["img"].astype(str) == str(img)].tolist()
            for idx in matches:
                if idx not in seen:
                    ordered_idxs.append(idx)
                    seen.add(idx)
        for idx in df.index.tolist():
            if idx not in seen:
                ordered_idxs.append(idx)
                seen.add(idx)
    else:
        ordered_idxs = df.index.tolist()

    if not ordered_idxs:
        return []

    df_ord = df.loc[ordered_idxs].reset_index(drop=True)
    imgs = df_ord["img"].astype(str).tolist()
    return imgs


def annotate_shared_plot(
    shared: dict,
    fontsize: int = 7,
    alpha: float = 0.25,
):
    """
    Use shared['img_order_labels']  to label the x-axis by image blocks
    and color the background for images listed in shared['high_list'] / shared['low_list'].
    Populates shared['image_legend_handles'] for the final legend.
    """
    ax = shared.get("ax")
    if ax is None:
        return

    # Colors
    high_color = "lightgreen"
    low_color = "lightcoral"
    neutral_color = None

    # Determine ordered image list and contiguous segments
    imgs = shared.get("img_order_labels", []) or []

    if not imgs:
        return

    # find contiguous segments of same image
    segments = []
    start = 0
    cur_img = imgs[0]
    for i in range(1, len(imgs)):
        if imgs[i] != cur_img:
            segments.append((cur_img, start, i))  # end exclusive
            start = i
            cur_img = imgs[i]
    segments.append((cur_img, start, len(imgs)))

    # ensure y placement for labels (near top of axis)
    ylim = ax.get_ylim()
    y_min, y_max = ylim[0], ylim[1]
    y_range = max(1e-6, y_max - y_min)
    label_y = y_max - 0.02 * y_range

    # shading and labels
    img_cat_map = shared.get("img_category_map", {}) or {}
    high_list = set(shared.get("high_list", []) or [])
    low_list = set(shared.get("low_list", []) or [])

    image_legend_handles = []
    added_high = False
    added_low = False

    for img_id, s, e in segments:
        # choose color
        col = None
        if img_id in high_list:
            col = high_color
            if not added_high:
                image_legend_handles.append(
                    Patch(facecolor=high_color, alpha=alpha, label=f"High {shared.get('emotion_type','')}".strip())
                )
                added_high = True
        elif img_id in low_list:
            col = low_color
            if not added_low:
                image_legend_handles.append(
                    Patch(facecolor=low_color, alpha=alpha, label=f"Low {shared.get('emotion_type','')}".strip())
                )
                added_low = True

        if col:
            # shade the background for this image block; align with data point centers
            ax.axvspan(s - 0.5, e - 0.5, facecolor=col, alpha=alpha, edgecolor=None)

        # label in the center of the segment: show img id and optionally category
        label = str(img_id)
        if img_cat_map and isinstance(img_cat_map, dict):
            cat = img_cat_map.get(img_id)
            if cat is not None:
                label = f"{img_id}\n{cat}"

        seg_len = e - s
        if seg_len >= 1:
            ax.text(
                (s + e - 1) / 2.0,
                label_y,
                label,
                ha="center",
                va="top",
                fontsize=fontsize,
                rotation=15,
                rotation_mode="anchor",
            )

    # store legend handles in shared for finalize_shared_plot to use
    if image_legend_handles:
        shared["image_legend_handles"] = image_legend_handles
    else:
        shared["image_legend_handles"] = []


def finalize_shared_plot(
    shared: dict,
    save_plot: bool = True,
    filename: str = "combined_vda_plot.png",
    bbox_inches="tight",
    dpi=300,
    show: bool = True,
):
    """
    Finalize shared plot: add image annotations (shading & labels), legend, save and show.
    """
    fig = shared["fig"]
    ax = shared["ax"]
    mode = shared.get("mode", "both")
    connect_segments = shared.get("connect_segments", False)

    # attempt to auto-set y-limits based on plotted lines
    try:
        ys = []
        for line in ax.get_lines():
            yd = np.asarray(line.get_ydata())
            ys.extend(yd[np.isfinite(yd)])
        if ys:
            y_min, y_max = np.min(ys), np.max(ys)
            y_range = max(1e-6, y_max - y_min)
            ax.set_ylim(y_min - 0.12 * y_range, y_max + 0.06 * y_range)
    except Exception:
        pass

    # --- build 2D matrix and plot group mean ---
    all_y_list = shared.get("all_y_values", [])
    if all_y_list and mode in ("mean", "both"):
        # assumes all y have same length
        y_mat = np.vstack(all_y_list)  # shape: (n_people, n_time)
        y_mean = np.nanmean(y_mat, axis=0)  # per-instant mean
        n = y_mean.shape[0]
        x = np.arange(y_mean.shape[0])
        mean_line = None
        if connect_segments:
            (mean_line,) = ax.plot(
                x,
                y_mean,
                color="black",
                linewidth=2.0,
                linestyle="-",
                alpha=0.9,
                label="Group mean",
            )
        else:
            # segmented mean line using global img_order_labels
            imgs = shared.get("img_order_labels", [])

            if not imgs:
                imgs = [None] * n  # fallback: just one continuous segment
            # build contiguous segments of same image
            segments = []
            start = 0
            cur_img = imgs[0] if imgs else None
            for i in range(1, n):
                if imgs[i] != cur_img:
                    segments.append((start, i))
                    start = i
                    cur_img = imgs[i]
            if n > 0:
                segments.append((start, n))

            for s, e in segments:
                seg_x = np.arange(s, e)
                seg_y = y_mean[s:e]
                if len(seg_x) == 1:
                    (lobj,) = ax.plot(
                        seg_x,
                        seg_y,
                        marker="o",
                        color="black",
                        markersize=4,
                        linestyle="None",
                        alpha=0.9,
                        label="Group mean" if mean_line is None else "_nolegend_",
                    )
                else:
                    (lobj,) = ax.plot(
                        seg_x,
                        seg_y,
                        color="black",
                        linewidth=2.0,
                        linestyle="-",
                        alpha=0.9,
                        label="Group mean" if mean_line is None else "_nolegend_",
                    )
                if mean_line is None:
                    mean_line = lobj
        if mean_line is not None:
            shared["legend_handles"].append(mean_line)
            shared["labels"].append("Group mean")

    ax.set_xlabel("Time Points", fontweight="bold")
    annotate_shared_plot(shared, fontsize=7, alpha=0.25)

    # combine legend handles (people + image legend handles)
    handles = shared.get("legend_handles", []) + shared.get("image_legend_handles", [])
    labels = (
        shared.get("labels", []) + [h.get_label() for h in shared.get("image_legend_handles", [])]
        if shared.get("image_legend_handles")
        else shared.get("labels", [])
    )
    if handles:
        ax.legend(
            handles=handles, labels=labels, loc="upper right", fontsize=9, framealpha=0.9, bbox_to_anchor=(1.15, 1)
        )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    if save_plot:
        os.makedirs(shared["output_dir"], exist_ok=True)
        outpath = os.path.join(shared["output_dir"], filename)
        fig.savefig(outpath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Combined plot saved to {outpath}")
    if show:
        plt.show()
    plt.close(fig)
    return


def main():
    """Main function to process EEG data"""
    # Populate `people` with folder names found in sub_data
    people = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    people = [p for p in people if p.startswith("E")]
    # gender = ["F" if int(p[1:]) % 2 == 0 else "M" for p in people]

    img_info = load_image_info()
    oasis_categories = img_info["oasis_categories"]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # remove existing directory if it exists
    os.mkdir(output_dir)

    vda = {}
    vda_results = {}
    va_seq = []
    ar_seq = []

    va_df = {}
    ar_df = {}
    va_vda = {}
    ar_vda = {}

    for person in people:
        filename = rf"{input_dir}\{person}\pow.csv"

        # Load data
        print(f"Loading EEG data from {person}")
        df = load_eeg_data(filename)
        df = clean_slice_df(df)

        # TODO: nicer normailzation
        # base_prev = compute_prev_R2_baseline(df)
        # df = apply_baseline_percent_change(df, base_prev_r2=base_prev)

        df = drop_first_image(df)

        # Keep only image presentation periods (slice "I")
        df_i = df[df["slice"] == "I"].reset_index(drop=True)
        print(f"Data loaded successfully. Shape: {df_i.shape}")

        # Calculate valence, dominance, and arousal
        print("\nCalculating valence, dominance, and arousal...")
        asym = calculate_asymetryies(df_i, df)
        va = calculate_valence(df_i, asym)
        ar = calculate_arousal(df_i, asym)
        dom = calculate_dominance(df_i, asym)
        vda[person] = va | ar | dom

        # Prepare VDA results for saving
        # Build a time-ordered DataFrame (one row per sample in df_i) and
        # expand any per-image summaries to per-row via the 'img' column.
        vda_df = pd.DataFrame()
        vda_df["img"] = df_i["img"].reset_index(drop=True)
        n_rows = len(vda_df)

        for key, val in vda.items():
            # Series (per-row) already aligned to time order -> reset_index to keep order
            if len(val) == n_rows:
                vda_df[key] = val.reset_index(drop=True)
            else:
                # per-image aggregated series (index = img) -> map to each row by img id
                vda_df[key] = vda_df["img"].map(val)
        vda_results[person] = vda_df

        va_df[person], va_vda[person], va_seq = filter_and_alternate_images(
            df_i,
            vda[person],
            oasis_categories,
            va_seq,
            emot_type="valence",
        )

        ar_df[person], ar_vda[person], ar_seq = filter_and_alternate_images(
            df_i,
            vda[person],
            oasis_categories,
            ar_seq,
            emot_type="arousal",
        )

    shared_plot_valence = init_shared_plot(
        img_seq=va_seq,
        img_category_map=oasis_categories,
        calc_method="valence",
        emotion_type="valence",
        output_dir=output_dir,
        mode="mean",
    )

    for person in people:
        add_person_to_shared_plot(
            person,
            va_df[person],
            va_vda[person],
            shared_plot_valence,
            connect_segments=False,
        )

    finalize_shared_plot(
        shared_plot_valence,
        filename="combined_valence_plot.png",
    )

    # plot_valence_arousal_methods(vda_results, bin_size=20, output_dir=output_dir)

    # Combine all VDA results into a single DataFrame for saving
    vda_df = pd.DataFrame()
    for person, res_df in vda_results.items():
        res_df = res_df.copy()
        res_df["person"] = person
        vda_df = pd.concat([vda_df, res_df], ignore_index=True)
    vda_df.to_csv(f"{output_dir}/VDA_results.csv", index=False)
    return


if __name__ == "__main__":
    main()
