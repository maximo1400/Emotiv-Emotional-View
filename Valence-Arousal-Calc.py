import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os, shutil
import pickle

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


def clean_slice_df(df: pd.DataFrame, time_s: int = EEG_CONFIG["time_s"]) -> pd.DataFrame:
    """Clean the DataFrame by removing rows with NaN or infinite values and balance image groups."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    # compute per-image counts once
    counts = df.groupby("img").size()
    min_count = int(counts.min())

    # take the last `min_count` rows per group
    balanced = df.groupby("img", group_keys=False).apply(lambda g: g.tail(min_count))

    balanced["slice"] = pd.NA
    # I1 = min_count * 1 // time_s
    # I2 = min_count * 4 // time_s
    # I3 = min_count * 6 // time_s
    # R1 = min_count * 8 // time_s
    # R2 = min_count
    # for img in balanced["img"].unique():
    #     indices = balanced[balanced["img"] == img].index
    #     balanced.loc[indices[:I1], "slice"] = "I1"
    #     balanced.loc[indices[I1:I2], "slice"] = "I2"
    #     balanced.loc[indices[I2:I3], "slice"] = "I3"
    #     balanced.loc[indices[I3:R1], "slice"] = "R1"
    #     balanced.loc[indices[R1:R2], "slice"] = "R2"
    I = min_count * 6 // time_s
    R1 = min_count * 8 // time_s
    R2 = min_count
    for img in balanced["img"].unique():
        indices = balanced[balanced["img"] == img].index
        balanced.loc[indices[:I], "slice"] = "I"
        balanced.loc[indices[I:R1], "slice"] = "R1"
        balanced.loc[indices[R1:R2], "slice"] = "R2"

    return balanced.reset_index(drop=True)


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
    results = {}
    # asymmetries = calculate_asymetryies(df)

    # Calculate final metrics with multiple approaches
    valence_methods = []

    # Valence from different alpha asymmetry methods
    valence_methods.append(-asymmetries["frontal_asym_alpha"])  # Standard log
    valence_methods.append(-asymmetries["frontal_asym_norm_alpha"])  # Normalized log

    # For ratio methods, convert to log scale for consistency
    ratio_raw = asymmetries["frontal_asym_ratio_alpha"]
    ratio_norm = asymmetries["frontal_asym_ratio_norm_alpha"]
    valence_methods.append(-np.log(ratio_raw + eps))  # Ratio as log
    valence_methods.append(-np.log(ratio_norm + eps))

    # Create composite scores (weighted average)
    if len(valence_methods) == 4:
        # Weight: standard=0.3, normalized=0.3, ratio=0.2, ratio_norm=0.2
        weights = [0.3, 0.3, 0.2, 0.2]
        results["valence"] = np.average(valence_methods, weights=weights, axis=0)
    else:
        print(f"Warning: Expected 4 valence methods, got {len(valence_methods)}")
        results["valence"] = np.mean(valence_methods, axis=0)

    # Store individual methods for comparison
    method_names = asymmetries["methods"]
    for i, method in enumerate(method_names[: len(valence_methods)]):
        results[f"valence_{method}"] = valence_methods[i]

    return results


def calculate_arousal(df: pd.DataFrame, asymmetries: dict, eeg_config=EEG_CONFIG, eps: float = 1e-10) -> dict:
    results = {}

    # Enhanced activation calculation
    frontal_electrodes = eeg_config["frontal_electrodes"]

    # Multiple activation measures
    beta_low_activities = []
    beta_high_activities = []
    beta_combined_activities = []

    for electrode in frontal_electrodes:
        beta_low_col = f"{electrode}/betaL"
        beta_high_col = f"{electrode}/betaH"

        beta_low_activities.append(df[beta_low_col])
        beta_high_activities.append(df[beta_high_col])
        beta_combined_activities.append(df[beta_low_col] + df[beta_high_col])

    # Store different activation measures
    results["activation_beta_low"] = np.mean(beta_low_activities, axis=0)
    results["activation_beta_high"] = np.mean(beta_high_activities, axis=0)
    results["activation_beta_combined"] = np.mean(beta_combined_activities, axis=0)

    # Primary activation measure (combined beta for robustness)
    results["activation"] = results["activation_beta_combined"]

    # Activation by reversed frontal log

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

    # Dominance calculation
    results["dominance"] = np.mean(dominance_methods, axis=0)

    # Store individual methods for comparison
    method_names = asymmetries["methods"]

    # Store individual methods
    for i, method in enumerate(method_names[: len(dominance_methods)]):
        results[f"dominance_{method}"] = dominance_methods[i]

    return results


def plot_valence_activation(
    vda_results,
    save_plot=True,
    dominance_is_size=True,
    bin_size=1,
    output_dir=output_dir,
):
    """
    Create a scatter plot of valence vs activation with quadrant analysis
    """
    if "valence" not in vda_results or "activation" not in vda_results or "dominance" not in vda_results:
        print("Error: Valence, activation, or dominance data not available for plotting")
        return

    valence = vda_results["valence"]
    activation = vda_results["activation"]
    dominance = vda_results["dominance"]

    n_points = len(valence)
    n_bins = n_points // bin_size

    valence_binned = []
    activation_binned = []
    dominance_binned = []

    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = start_idx + bin_size

        valence_binned.append(np.mean(valence[start_idx:end_idx]))
        activation_binned.append(np.mean(activation[start_idx:end_idx]))
        dominance_binned.append(np.mean(dominance[start_idx:end_idx]))

    # Convert to numpy arrays
    valence = np.array(valence_binned)
    activation = np.array(activation_binned)
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
        activation,
        alpha=0.7,
        s=sizes,
        c="blue",
        edgecolors="black",
        linewidth=0.5,
    )

    # Add quadrant lines
    plt.axhline(
        y=np.mean(activation),
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
    y_absmax = max(abs(np.min(activation)), abs(np.max(activation)), 0.1) * 1.1
    axis_limit = max(x_absmax, y_absmax)
    plt.xlim(-axis_limit, axis_limit)
    plt.ylim(-axis_limit, axis_limit)

    # Get current axis limits
    x_range = plt.xlim()
    y_range = plt.ylim()

    plt.text(
        x_range[1] * 0.8,
        y_range[1] * 0.9,
        "High Activation\nPositive Valence\n(Happy/Excited)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[1] * 0.9,
        "High Activation\nNegative Valence\n(Angry/Stressed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    plt.text(
        x_range[1] * 0.8,
        y_range[0] * 0.8,
        "Low Activation\nPositive Valence\n(Calm/Relaxed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[0] * 0.8,
        "Low Activation\nNegative Valence\n(Sad/Depressed)",
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    # Customize the plot
    plt.xlabel("Valence (Negative ← → Positive)", fontsize=12, fontweight="bold")
    plt.ylabel("Activation (Low ← → High)", fontsize=12, fontweight="bold")
    plt.title(
        "EEG-based Emotional State: Valence vs Activation",
        fontsize=14,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Data Points: {len(valence)}\n"
    stats_text += f"Valence: μ={np.mean(valence):.3f}, σ={np.std(valence):.3f}\n"
    stats_text += f"Activation: μ={np.mean(activation):.3f}, σ={np.std(activation):.3f}\n"
    stats_text += f"Dominance: μ={np.mean(dominance):.3f}, σ={np.std(dominance):.3f}\n"

    # Add quadrant counts
    q1 = np.sum((valence >= 0) & (activation >= 0))  # Happy
    q2 = np.sum((valence < 0) & (activation >= 0))  # Angry
    q3 = np.sum((valence < 0) & (activation < 0))  # Sad
    q4 = np.sum((valence >= 0) & (activation < 0))  # Calm

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
        plt.savefig(f"{output_dir}/valence_activation_plot.png", dpi=300, bbox_inches="tight")
        print("Plot saved as 'valence_activation_plot.png'")

    plt.show()


def plot_time_series(vda_results, save_plot=True, output_dir=output_dir):
    """
    Create time series plots for valence, activation, and dominance
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    metrics = ["valence", "activation", "dominance"]
    colors = ["blue", "red", "green"]
    labels = [
        "Valence (Negative ← → Positive)",
        "Activation (Low ← → High)",
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


def plot_valence_activation_methods(
    vda_results,
    save_plot=True,
    dominance_is_size=True,
    bin_size=1,
    output_dir=output_dir,
):
    """
    Create a scatter plot showing all valence and activation calculation methods with different colors
    """
    # Define the methods we want to plot
    # valence_methods = ["valence_standard", "valence_normalized", "valence_ratio", "valence_ratio_norm"]
    valence_methods = ["valence_normalized", "valence_ratio_norm", "valence"]
    # activation_methods = ["activation_beta_low", "activation_beta_high", "activation_beta_combined"]
    activation_methods = ["activation_beta_low", "activation"]

    # Check if methods exist in results
    available_valence = [method for method in valence_methods if method in vda_results]
    available_activation = [method for method in activation_methods if method in vda_results]

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
    all_activation = []
    custom_legend_elements = []

    for v_method in available_valence:
        for a_method in available_activation:
            if plot_idx >= len(colors):
                break

            # Get data
            valence_data = vda_results[v_method]
            activation_data = vda_results[a_method]

            # Apply binning
            valence_binned = []
            activation_binned = []

            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = start_idx + bin_size
                valence_binned.append(np.mean(valence_data[start_idx:end_idx]))
                activation_binned.append(np.mean(activation_data[start_idx:end_idx]))

            valence_binned = np.array(valence_binned)
            activation_binned = np.array(activation_binned)

            # Store for axis limits calculation
            all_valence.extend(valence_binned)
            all_activation.extend(activation_binned)

            # Create scatter plot
            color = colors[plot_idx]
            marker = markers[plot_idx % len(markers)]

            # Create cleaner label
            v_label = v_method.replace("valence_", "").replace("_", " ").title()
            a_label = a_method.replace("activation_", "").replace("_", " ").title()
            label = f"{v_label} + {a_label}"

            plt.scatter(
                valence_binned,
                activation_binned,
                alpha=0.6,
                s=sizes,
                c=color,
                marker=marker,
                edgecolors="black",
                linewidth=0.3,
                # label=f"{v_method.replace('valence_', 'V:')} + {a_method.replace('activation_', 'A:')}",
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
    all_activation = np.array(all_activation)

    # Add reference lines
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=2, label="Zero line")
    plt.axvline(x=0, color="black", linestyle="-", alpha=0.8, linewidth=2)
    plt.axhline(
        y=np.mean(all_activation),
        color="red",
        linestyle="--",
        alpha=0.6,
        linewidth=1,
        label="Overall mean",
    )
    plt.axvline(x=np.mean(all_valence), color="red", linestyle="--", alpha=0.6, linewidth=1)

    # Force (0,0) as the visual center
    x_absmax = max(abs(np.min(all_valence)), abs(np.max(all_valence)), 0.1) * 1.1
    y_absmax = max(abs(np.min(all_activation)), abs(np.max(all_activation)), 0.1) * 1.1
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
        "High Activation\nPositive Valence\n(Happy/Excited)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[1] * 0.9,
        "High Activation\nNegative Valence\n(Angry/Stressed)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7),
    )

    plt.text(
        x_range[1] * 0.8,
        y_range[0] * 0.8,
        "Low Activation\nPositive Valence\n(Calm/Relaxed)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
    )

    plt.text(
        x_range[0] * 0.8,
        y_range[0] * 0.8,
        "Low Activation\nNegative Valence\n(Sad/Depressed)",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    # Customize the plot
    plt.xlabel("Valence (Negative ← → Positive)", fontsize=12, fontweight="bold")
    plt.ylabel("Activation (Low ← → High)", fontsize=12, fontweight="bold")
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
    stats_text += f"Activation range: [{np.min(all_activation):.3f}, {np.max(all_activation):.3f}]\n"
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
            f"{output_dir}/valence_activation_methods_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        print("Plot saved as 'valence_activation_methods_comparison.png'")

    plt.show()

    # Print method summary
    print("\nMethod combinations plotted:")
    plot_idx = 0
    for v_method in available_valence:
        for a_method in available_activation:
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


def main():
    """Main function to process EEG data"""
    # Populate `people` with folder names found in sub_data
    people = [name for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
    people = [p for p in people if p.startswith("E")]

    img_info = load_image_info()
    oasis_categories = img_info["oasis_categories"]

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # remove existing directory if it exists
    os.mkdir(output_dir)

    # for person in people:
    person = people[0]
    filename = rf"{input_dir}\{person}\pow.csv"

    # Load data
    print(f"Loading EEG data from {person}")
    df = load_eeg_data(filename)
    df = clean_slice_df(df)
    # df.to_csv("data.csv", index=False)
    # return

    # TODO: nicer normailzation
    # base_prev = compute_prev_R2_baseline(df)
    # df = apply_baseline_percent_change(df, base_prev_r2=base_prev)

    df = drop_first_image(df)
    print(f"Data loaded successfully. Shape: {df.shape}")

    # filter rest periods and first 3 seconds (Only keep I2)
    # df_i2 = df[df["slice"] == "I2"].reset_index(drop=True)
    df_i = df[df["slice"] == "I"].reset_index(drop=True)
    # df_i2 = df.copy()

    # Calculate valence, dominance, and activation
    print("\nCalculating valence, dominance, and activation...")
    asym = calculate_asymetryies(df_i, df)
    va = calculate_valence(df_i, asym)
    ar = calculate_arousal(df_i, asym)
    dom = calculate_dominance(df_i, asym)

    # Save results to CSV
    print("\nSaving results...")

    vda = va | ar | dom
    # for key, val in vda.items():
    #     print(all(vda2[key] == vda[key]))

    # Prepare VDA results for saving
    vda_df_data = {}
    for key, values in vda.items():
        vda_df_data[key] = values

    vda_df = pd.DataFrame(vda_df_data)
    # vda_df.to_csv(f"{output_dir}/VDA_results.csv", index=False)
    print(f"VDA results saved to '{output_dir}/VDA_results.csv'")

    # plot_valence_activation(vda_results, output_dir=output_dir)
    plot_time_series(vda, output_dir=output_dir)
    # plot_valence_activation_methods(vda_results, bin_size=20, output_dir=output_dir)

    return


if __name__ == "__main__":
    main()
