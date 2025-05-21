import os
import warnings

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import shap
from IPython.core.display import HTML

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn


def get_feature_names(test_set, modalities):
    """
    Get feature names based on the modality type.

    Parameters:
    - test_set: The test set object containing feature information.
    - modality_type: Type of data ('tabular', 'ts-vitals', 'ts-labs' or 'notes').

    Returns:
    - feature_names: List of feature names.
    """
    fn_map = {}
    for modality_type in modalities:
        if modality_type == "static":
            feature_names = test_set.get_feature_list()
            fn_map['static'] = feature_names
        elif modality_type == "timeseries":
            #print(test_set.col_dict)
            feature_names = test_set.get_feature_list("dynamic0")
            fn_map['ts-vitals'] = feature_names
            feature_names = test_set.get_feature_list("dynamic1")
            fn_map['ts-labs'] = feature_names

    return fn_map

class ModelWrapper(torch.nn.Module):
    """
    A wrapper around the model to ensure scalar outputs for SHAP.
    """
    def __init__(self, model, modality,
                 total_dim, target_size,
                 ts_ind=None):
        super().__init__()
        self.model = model
        self.modality = modality
        self.ts_ind = ts_ind  # Index for timeseries data
        self.fc = nn.Linear(total_dim, target_size)  # Linear layer for final output

    def forward(self, x):
        if self.modality == "static":
            # Forward pass for static modality
            embed = self.model.embed_static(x)
        elif self.modality == "timeseries":
            # Forward pass for timeseries modality
            self.model.train()
            embed = self.model.embed_timeseries[self.ts_ind](x)
            embed = embed.view(embed.size(0), embed.size(1), 1)
        elif self.modality == "notes":
            # Forward pass for notes modality
            embed = self.model.embed_notes(x)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")

        # Pass through the Linear layer to generate outputs
        if self.modality == "timeseries":
            self.model.eval()
            return embed
        return self.fc(embed)

def get_shap_values(model, 
                    batch, 
                    device,
                    num_ts,
                    modalities):
    """
    Compute SHAP values for each modality using the model's prepare_batch method.
    """
    s, d, lengths, n = batch[0], batch[2], batch[3], batch[4]
    ts_data = {}
    for i in range(num_ts):
        ts_data['dynamic' + str(i)] = d[i]
    shap_values = {}

    if "static" in modalities:
        wrapper_static = ModelWrapper(model, "static", total_dim=64, target_size=1).to(device)
        explainer_static = shap.DeepExplainer(wrapper_static, s.to(device))
        shap_values["static"] = explainer_static.shap_values(s.to(device), check_additivity=False)
        shap_values["static_expected"] = explainer_static.expected_value[0]
    if "timeseries" in modalities:
        ts_shap_values = {}
        for i in range(num_ts):
            wrapper_ts = ModelWrapper(model, "timeseries", total_dim=128, target_size=1, ts_ind=i).to(device)
            explainer_ts = shap.DeepExplainer(wrapper_ts, ts_data['dynamic' + str(i)].to(device))
            ts_shap_values['dynamic' + str(i)] = explainer_ts.shap_values(ts_data['dynamic' + str(i)].to(device), check_additivity=False)
            ts_shap_values['dynamic' + str(i) + "_expected"] = np.array([np.mean(explainer_ts.expected_value)])
        shap_values["timeseries"] = ts_shap_values
    if "notes" in modalities:
        wrapper_notes = ModelWrapper(model, "notes", total_dim=64, target_size=1).to(device)
        explainer_notes = shap.DeepExplainer(wrapper_notes, n.to(device))
        shap_values["notes"] = explainer_notes.shap_values(n.to(device), check_additivity=False)
        shap_values["notes_expected"] = explainer_notes.expected_value[0]

    return shap_values

def estimate_mm_summary(shap_scores, shap_expected_scores):
    # Average expected values across modalities
    shap_expected_ovr = np.mean([shap_expected_scores[0], 
                                 shap_expected_scores[1], 
                                 shap_expected_scores[2], 
                                 shap_expected_scores[3]], 
                                axis=0).round(3)
    # Merge all four arrays into one list to get a unified range
    merged_shap = np.concatenate([shap_scores[0], shap_scores[1], 
                                  shap_scores[2], shap_scores[3]])
    # Get max and min SHAP values for valid range
    shap_max_ovr = np.max(merged_shap).round(5)
    shap_max_ovr = round(max([shap_max_ovr, shap_expected_ovr]), 5) + 0.01
    shap_min_ovr = np.min(merged_shap).round(5)
    shap_min_ovr = round(min([shap_min_ovr, shap_expected_ovr]), 5) - 0.01
    # Estimate absolute sum of SHAP values for each modality
    shap_sum_static = np.sum(np.abs(shap_scores[0]))
    shap_sum_ts = np.sum(np.abs(shap_scores[1])) + np.sum(np.abs(shap_scores[2]))
    shap_sum_notes = np.sum(np.abs(shap_scores[3]))
    shap_denom = shap_sum_static + shap_sum_ts + shap_sum_notes
    # Estimate multimodal degrees of importance (MM-SHAP)
    shap_static_degree = round((shap_sum_static / shap_denom) * 100, 2)
    shap_ts_degree = round((shap_sum_ts / shap_denom) * 100, 2)
    shap_notes_degree = round((shap_sum_notes / shap_denom) * 100, 2)
    shap_mm_scores = [shap_static_degree, shap_ts_degree, shap_notes_degree]

    return shap_mm_scores, shap_expected_ovr, shap_max_ovr, shap_min_ovr


def get_shap_summary_plot(
    shap_obj,
    outcome="In-hospital Death",
    fusion_type=None,
    modality=None,
    max_features=20,
    figsize=(7, 8),
    save_path=None,
    heatmap=False,
):
    """
    Generate a SHAP summary plot.

    Parameters:
    - shap_values: The calculated SHAP values.
    - feature_names: List of feature names.
    - title: Title for the plot (optional).

    Returns:
    - None: Displays the SHAP summary plot.
    """
    plt.figure()
    shap.initjs()
    if modality in ["static", "timeseries"]:
        if heatmap:
            shap_obj.values = shap_obj.values.round(3)
            shap.plots.heatmap(
                shap_obj,
                max_display=max_features,
                plot_width=9,
                show=False
            )
        else:
            shap.plots.beeswarm(
                shap_obj,
                plot_size=figsize,
                max_display=max_features,
                show=False,
            )
    if modality == "notes":
        shap.plots.bar(
            shap_obj,
            max_display=max_features,
            show=False,
        )
    
    if modality != "notes":
        plt.grid('both', linestyle='--', alpha=0.7)
    if heatmap:
        plt.title(f"Heatmap view for {modality} modality.")
    else:
        plt.title(f"SHAP Global Importance for {modality} modality: {outcome}, {fusion_type}.")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"SHAP summary plot for {modality} modality saved to {save_path}.")

def aggregate_ts(data):
    """
    Use mean aggregation to average timeseries data from the final hidden layer.
    Ignores missing values from the model output.

    Parameters:
    - shap_values: The calculated SHAP values.

    Returns:
    - aggregated_shap_values: Dictionary with feature names as keys and aggregated SHAP values as values.
    """
    to_agg = []
    ## Remove 0-padded intervals
    for ts in range(len(data)):
        ## Check if all values in data[ts] are 0
        if np.all(data[ts] == 0):
            continue
        to_agg.append(data[ts])

    ## Do not aggregate on missing values
    to_agg = np.array(to_agg)
    # Mask -1 values
    mask = to_agg > -1
    # Set -1 values to np.nan for mean calculation
    to_agg_masked = np.where(mask, to_agg, np.nan)
    # Suppress mean of empty slice warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        agg_values = np.nanmean(to_agg_masked, axis=0, keepdims=True)
    agg_values = np.abs(agg_values.reshape(1, -1))
    agg_values = np.where(np.isnan(agg_values), -1, agg_values).round(2)
    return agg_values


def get_shap_local_decision_plot(
    shap_obj,
    risk_quantile=None,
    figsize=(6, 7),
    save_static_path=None,
    save_ts0_path=None,
    save_ts1_path=None,
    save_nt_path=None,
    shap_range=None,
    mm_scores=None,
):
    """
    Generate a SHAP decision plot for tabular and timeseries data.
    Generate a SHAP text plot for notes data.
    Estimate and show the multimodal degree of dependence.
    """
    shap.initjs()
    # Set format for SHAP values
    for i in range(len(shap_obj)):
        #shap_obj[i].values = shap_obj[i].values.round(6)
        shap_obj[i].base_values = shap_obj[i].base_values.round(3)
        if i in [1, 2]:
            shap_obj[i].data = shap_obj[i].data.round(2)
        #print(max(shap_obj[i][0].values.round(3)), min(shap_obj[i][0].values.round(3)))
        ### Static EHR modality
        if i == 0:
            fig = plt.figure(figsize=figsize)
            shap_obj[i].data = np.where(shap_obj[i].data == 0, 'No', shap_obj[i].data)
            shap_obj[i].data = np.where(shap_obj[i].data == '1', 'Yes', shap_obj[i].data)
            ## Static EHR modality
            shap.plots.decision(
                shap_obj[0].base_values,
                shap_obj[0].values,
                shap_obj[0].data,
                shap_obj[0].feature_names,
                feature_display_range=slice(None, -16, -1),
                highlight=0,
                show=False,
                xlim=shap_range,
            )
            plt.title(f"Static EHR modality (RQ={risk_quantile}, TAB-SHAP={mm_scores[0]}%).")
            plt.tight_layout()
            plt.savefig(save_static_path, dpi=300)
            plt.close()
        ### Timeseries modalities
        if i in [1, 2]:
            shap_obj[i].data = np.where(shap_obj[i].data == -1, 'Missing', shap_obj[i].data)
            fig = plt.figure(figsize=figsize)
            shap.plots.decision(
                    shap_obj[i].base_values,
                    shap_obj[i].values,
                    shap_obj[i].data,
                    shap_obj[i].feature_names,
                    feature_display_range=slice(None, -11, -1),
                    highlight=0,
                    show=False,
                    xlim=shap_range,
                )
            if i == 1:
                plt.title(f"TS Vitals modality (RQ={risk_quantile}, TS-SHAP={mm_scores[1]}%).")
            else:
                plt.title(f"TS Labs modality (RQ={risk_quantile}, TS-SHAP={mm_scores[1]}%).")
                
            plt.tight_layout()
            if i == 1:
                plt.savefig(save_ts0_path, bbox_inches="tight", dpi=300)
            else:
                plt.savefig(save_ts1_path, bbox_inches="tight", dpi=300)
            plt.close()
        ### Notes modality
        if i == 3:
            shap_obj[i].values = np.array([shap_obj[i].values])[0]
            #shap_obj[i].base_values = shap_obj[i].base_values[0]
            shap_obj[i].data = shap_obj[i].data.astype('O')
            plot_highlighted_text_with_colorbar(shap_obj[i].values.round(3),
                                                shap_obj[i].data,
                                                shap_obj[i].base_values,
                                                mm_scores[2],
                                                save_path=save_nt_path,
                                                shap_range=shap_range)

    print(f"SHAP local-level decision plots saved to disk.")

def plot_highlighted_text_with_colorbar(shap_values, text_tokens, 
                                        expected_value, mm_score,
                                        figsize=(10, 4), 
                                        cmap="coolwarm", save_path=None,
                                        shap_range=None):
    """
    Display a highlighted text plot and a colorbar based on SHAP values.

    Args:
        shap_values (np.ndarray): SHAP values for each token (1D array).
        text_tokens (list of str): List of text tokens (words or sentences).
        expected_value (float): Reference value for colorbar label.
        figsize (tuple): Figure size.
        cmap (str): Matplotlib colormap.
        save_path (str): If provided, saves the plot to this path.
    """
    norm = mpl.colors.Normalize(vmin=shap_range[0], vmax=shap_range[1])
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    # Prepare text with colored background, handling new lines every 12 words and after every '#' symbol
    x = 0.01
    y = 0.95
    line_height = 0.08
    max_x = 0.98
    for token, val in zip(text_tokens, shap_values):
        words = token.split()
        word_count = 0
        token_line = ""
        for word in words:
            token_line += word + " "
            word_count += 1
            # New line every 15 words
            if word_count % 15 == 0:
                # Insert note separator if needed
                if "<ENDNOTE> <STARTNOTE>" in token_line:
                    parts = token_line.split("<ENDNOTE> <STARTNOTE>")
                    for idx, part in enumerate(parts):
                        if part.strip():
                            color = cmap(norm(val))
                            text_width = 0.01 * len(part)
                            if x + text_width > max_x:
                                x = 0.01
                                y -= line_height
                            ax.text(x, y, part, fontsize=10, va='top', ha='left',
                                    bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.1'))
                            x += text_width
                        if idx < len(parts) - 1:
                            # Add separator and move to new line
                            x = 0.01
                            y -= line_height
                            ax.text(x, y, "------------NEXT NOTE-------------", fontsize=11, va='top', ha='left', color='black')
                            y -= line_height
                            x = 0.01
                    token_line = ""
                else:
                    color = cmap(norm(val))
                    text_width = 0.01 * len(token_line)
                    if x + text_width > max_x:
                        x = 0.01
                        y -= line_height
                    ax.text(x, y, token_line, fontsize=10, va='top', ha='left',
                            bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.1'))
                    x = 0.01
                    y -= line_height
                    token_line = ""
            # New line after every '#' symbol in word
            if '#' in word:
                color = cmap(norm(val))
                text_width = 0.01 * len(token_line)
                if x + text_width > max_x:
                    x = 0.01
                    y -= line_height
                ax.text(x, y, token_line, fontsize=10, va='top', ha='left',
                        bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.1'))
                x = 0.01
                y -= line_height
                token_line = ""
        if token_line.strip():
            # Insert note separator if needed
            if "<ENDNOTE> <STARTNOTE>" in token_line:
                parts = token_line.split("<ENDNOTE> <STARTNOTE>")
                for idx, part in enumerate(parts):
                    if part.strip():
                        color = cmap(norm(val))
                        text_width = 0.01 * len(part + " ")
                        if x + text_width > max_x:
                            x = 0.01
                            y -= line_height
                        ax.text(x, y, part + " ", fontsize=10, va='top', ha='left',
                                bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.1'))
                        x += text_width
                    if idx < len(parts) - 1:
                        x = 0.01
                        y -= line_height
                        ax.text(x, y, "------------NEXT NOTE---------------", fontsize=11, va='top', ha='left', color='black')
                        y -= line_height
                        x = 0.01
            else:
                color = cmap(norm(val))
                text_width = 0.01 * len(token_line + " ")
                if x + text_width > max_x:
                    x = 0.01
                    y -= line_height
                ax.text(x, y, token_line + " ", fontsize=10, va='top', ha='left',
                        bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.1'))
                x += text_width
        else:
            x = 0.01

    # Draw a bounding box around the text area
    min_y = y - line_height
    rect = mpl.patches.FancyBboxPatch(
        (0, min_y), 1, 0.98-min_y,
        boxstyle="round,pad=0.01",
        linewidth=2,
        edgecolor='black',
        facecolor='none'
    )
    ax.add_patch(rect)

    # Add colorbar at the top and center it
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Set colorbar to cover the full plot width
    cbar = plt.colorbar(
        sm, ax=ax, orientation='horizontal',
        fraction=0.15, pad=0.02, anchor=(0.5, 1.0), location='top'
    )
    cbar.set_label(f'SHAP Values (Ref={expected_value:.3f}, T-SHAP={mm_score}%)', fontsize=14, labelpad=10)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('top')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
