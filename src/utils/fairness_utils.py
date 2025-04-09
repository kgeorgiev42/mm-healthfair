import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from functools import partial
from scipy.stats import norm

from tqdm import tqdm
from utils.functions import load_pickle

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_ratio,
    equalized_odds_ratio,
    equal_opportunity_ratio,
)

def plot_bar_metric_frame(
    metrics: dict,
    y_test: np.ndarray,
    y_hat: np.ndarray,
    ehr_static: pl.DataFrame,
    title: str,
    save_path: str,
    figsize=(10, 6),
    fontsize=14,
):
    """
    Plot a bar chart for the given metric frame.

    Args:
        metric_frame (MetricFrame): The metric frame to plot.
        title (str): The title of the plot.
        save_path (str): The path to save the plot.
        figsize (tuple): The size of the figure.
        fontsize (int): The font size for the plot.

    Returns:
        None
    """
    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_test,
        y_pred=y_hat,
        sensitive_features=ehr_static,
    )

    metric_frame.by_group.plot.bar(
        subplots=True,
        layout=[3, 2],
        colormap="Pastel2",
        legend=False,
        figsize=[8, 6],
        title="Fairness evaluation",
        xlabel="Attribute",
    )
    # Create a bar plot for the metric frame
    metric_frame.plot.bar(figsize=figsize, fontsize=fontsize)
    plt.title(title, fontsize=fontsize)
    plt.xlabel("Group", fontsize=fontsize)
    plt.ylabel("Metric Value", fontsize=fontsize)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)

# Extract bias-corrected confidence intervals using BCa method (example of effects and variations: https://www.erikdrysdale.com/bca_python/)
def bias_corrected_ci(bootstrap_samples, observed_value):
    sorted_samples = np.sort(bootstrap_samples)
    # Calculate z0 (bias correction factor)
    z0 = norm.ppf((np.sum(sorted_samples < observed_value) + 0.5) / len(sorted_samples))
    # Calculate a (acceleration factor) using jackknife
    jackknife_estimates = [np.mean(np.delete(sorted_samples, i)) for i in range(len(sorted_samples))]
    mean_jackknife = np.mean(jackknife_estimates)
    a = np.sum((mean_jackknife - jackknife_estimates) ** 3) / (
        6 * (np.sum((mean_jackknife - jackknife_estimates) ** 2) ** 1.5)
    )
    # Adjust percentiles
    alpha = [0.025, 0.975]  # For a 95% CI
    adjusted_percentiles = norm.cdf(z0 + (z0 + norm.ppf(alpha)) / (1 - a * (z0 + norm.ppf(alpha))))
    lower_bound = np.percentile(sorted_samples, adjusted_percentiles[0] * 100)
    upper_bound = np.percentile(sorted_samples, adjusted_percentiles[1] * 100)
    return lower_bound, upper_bound

def get_bootstrapped_fairness_measures(y_test: np.ndarray, 
                                       y_hat: np.ndarray, 
                                       attr_pf: pl.DataFrame,
                                       n_boot: int = 1000,
                                       seed: int = 0,
                                       verbose: bool = False) -> tuple:
    global_metrics = {
        "Demographic Parity": demographic_parity_ratio,
        "Equalized Odds": equalized_odds_ratio,
        "Equal Opportunity": equal_opportunity_ratio,
    }
    # Initialize random state
    rng = np.random.default_rng(seed)
    if verbose:
        print(attr_pf.to_pandas().value_counts(normalize=True))
    if 'race_group' in attr_pf.columns:
        attr_pf = attr_pf.filter(pl.col('race_group')!='Other')
    # Store bootstrap results
    bootstrap_results = {metric: [] for metric in global_metrics}
    # Perform bootstrapping (random sampling with stratification)
    for _ in tqdm(range(n_boot)):
        # Sample with stratification
        stratified_indices = []
        for group in np.unique(attr_pf.to_numpy()):
            group_indices = np.where(attr_pf.to_numpy() == group)[0]
            stratified_indices.extend(
                rng.choice(group_indices, size=len(group_indices), replace=True)
            )
        stratified_indices = np.array(stratified_indices)

        y_test_sample = y_test[stratified_indices]
        y_hat_sample = y_hat[stratified_indices]
        attr_pf_sample = attr_pf[stratified_indices, :]

        # Calculate metrics for the sample
        dp_boot = demographic_parity_ratio(y_test_sample, y_hat_sample, sensitive_features=attr_pf_sample)
        eor_boot = equalized_odds_ratio(y_test_sample, y_hat_sample, sensitive_features=attr_pf_sample)
        eop_boot = equal_opportunity_ratio(y_test_sample, y_hat_sample, sensitive_features=attr_pf_sample)
        # Replace 0 values with 0.001 to avoid division by zero in ratios
        eor_boot = max(eor_boot, 0.001)
        eop_boot = max(eop_boot, 0.001)

        # Store results
        bootstrap_results["Demographic Parity"].append(dp_boot)
        bootstrap_results["Equalized Odds"].append(eor_boot)
        bootstrap_results["Equal Opportunity"].append(eop_boot)

    # Calculate overall metrics from bootstrap results (under imbalance assumption)
    dpr = np.median(bootstrap_results["Demographic Parity"])
    eor = np.median(bootstrap_results["Equalized Odds"])
    eop = np.median(bootstrap_results["Equal Opportunity"])

    dp_ci = bias_corrected_ci(bootstrap_results["Demographic Parity"], dpr)
    eor_ci = bias_corrected_ci(bootstrap_results["Equalized Odds"], eor)
    eop_ci = bias_corrected_ci(bootstrap_results["Equal Opportunity"], eop)

    dpr_full = (dpr, dp_ci[0], dp_ci[1])
    eor_full = (eor, eor_ci[0], eor_ci[1])
    eop_full = (eop, eop_ci[0], eop_ci[1])

    return dpr_full, eor_full, eop_full
