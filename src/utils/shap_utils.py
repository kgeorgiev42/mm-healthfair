import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from torch import nn


def get_feature_names(test_set, modalities):
    """
    Get feature names for each modality in the test set and save the column mappings to a dictionary.

    Args:
        test_set (MIMIC4Dataset): The test set object containing feature information.
        modalities (list): List of modality types ('static', 'timeseries', 'notes').

    Returns:
        dict: Mapping from modality type to list of feature names.
    """
    fn_map = {}
    for modality_type in modalities:
        if modality_type == "static":
            feature_names = test_set.get_feature_list()
            fn_map["static"] = feature_names
        elif modality_type == "timeseries":
            feature_names = test_set.get_feature_list("dynamic0")
            fn_map["ts-vitals"] = feature_names
            feature_names = test_set.get_feature_list("dynamic1")
            fn_map["ts-labs"] = feature_names

    return fn_map


class ModelWrapper(torch.nn.Module):
    """
    A DeepSHAP wrapper around the PyTorch model to ensure scalar outputs for SHAP.

    Args:
        model: The base model to wrap.
        modality (str): The modality type ('static', 'timeseries', 'notes').
        total_dim (int): Input dimension for the linear layer.
        target_size (int): Output dimension for the linear layer.
        ts_ind (int, optional): Target for timeseries data (0 - vital signs, 1 - lab measurements).
    """

    def __init__(self, model, modality, total_dim, target_size, ts_ind=None):
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


def get_shap_values(model, batch, device, num_ts, modalities):
    """
    Batch-wise SHAP computation of attribution scores for each modality using the model's prepare_batch method.

    Args:
        model (MMModel): The model to explain.
        batch (DataLoader object): Batch of input data.
        device (torch.device): Torch device ('cpu' or 'gpu').
        num_ts (int): Number of timeseries modalities.
        modalities (list): List of modalities to explain.

    Returns:
        dict: SHAP values for each modality.
    """
    s, d, _, n = batch[0], batch[2], batch[3], batch[4]
    ts_data = {}
    for i in range(num_ts):
        ts_data["dynamic" + str(i)] = d[i]
    shap_values = {}

    if "static" in modalities:
        wrapper_static = ModelWrapper(model, "static", total_dim=64, target_size=1).to(
            device
        )
        explainer_static = shap.DeepExplainer(wrapper_static, s.to(device))
        shap_values["static"] = explainer_static.shap_values(
            s.to(device), check_additivity=False
        )
        shap_values["static_expected"] = explainer_static.expected_value[0]
    if "timeseries" in modalities:
        ts_shap_values = {}
        for i in range(num_ts):
            wrapper_ts = ModelWrapper(
                model, "timeseries", total_dim=128, target_size=1, ts_ind=i
            ).to(device)
            explainer_ts = shap.DeepExplainer(
                wrapper_ts, ts_data["dynamic" + str(i)].to(device)
            )
            ts_shap_values["dynamic" + str(i)] = explainer_ts.shap_values(
                ts_data["dynamic" + str(i)].to(device), check_additivity=False
            )
            ts_shap_values["dynamic" + str(i) + "_expected"] = np.array(
                [np.mean(explainer_ts.expected_value)]
            )
        shap_values["timeseries"] = ts_shap_values
    if "notes" in modalities:
        wrapper_notes = ModelWrapper(model, "notes", total_dim=64, target_size=1).to(
            device
        )
        explainer_notes = shap.DeepExplainer(wrapper_notes, n.to(device))
        shap_values["notes"] = explainer_notes.shap_values(
            n.to(device), check_additivity=False
        )
        shap_values["notes_expected"] = explainer_notes.expected_value[0]

    return shap_values


def estimate_mm_summary(shap_scores, shap_expected_scores):
    """
    Estimate aggregate multimodal SHAP values for calculating relative degree of modality dependence.
    Aggregate SHAP values are inspired by MM-SHAP (https://github.com/Heidelberg-NLP/MM-SHAP).

    Args:
        shap_scores (list): List of SHAP value arrays for each modality.
        shap_expected_scores (list): List of expected (reference) SHAP values for each modality. Estimated from the batch-wise SHAP mean.

    Returns:
        tuple: (mm_scores, shap_expected_ovr, shap_max_ovr, shap_min_ovr)
        mm_scores (list): Multimodal SHAP scores for each modality.
        shap_expected_ovr (float): Overall expected SHAP value across modalities.
        shap_max_ovr (float): Maximum SHAP value across all modalities.
        shap_min_ovr (float): Minimum SHAP value across all modalities.
    """
    # Average expected values across modalities
    shap_expected_ovr = np.mean(
        [
            shap_expected_scores[0],
            shap_expected_scores[1],
            shap_expected_scores[2],
            shap_expected_scores[3],
        ],
        axis=0,
    ).round(3)
    # Merge all four arrays into one list to get a unified range
    merged_shap = np.concatenate(
        [shap_scores[0], shap_scores[1], shap_scores[2], shap_scores[3]]
    )
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
    Generate and save a global-level SHAP summary plot for a given modality.

    Args:
        shap_obj: SHAP explanation object.
        outcome (str): Outcome name for plot title.
        fusion_type (str): Fusion type for plot title.
        modality (str): Modality type ('static', 'timeseries', 'notes').
        max_features (int): Maximum number of features to display.
        figsize (tuple): Figure size.
        save_path (str): Path to save the plot.
        heatmap (bool): If True, plot as heatmap.

    Returns:
        None
    """
    plt.figure()
    shap.initjs()
    if modality in ["static", "timeseries"]:
        if heatmap:
            shap_obj.values = shap_obj.values.round(3)
            shap.plots.heatmap(
                shap_obj, max_display=max_features, plot_width=9, show=False
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
        plt.grid("both", linestyle="--", alpha=0.7)
    if heatmap:
        plt.title(f"Heatmap view for {modality} modality.")
    else:
        plt.title(
            f"SHAP Global Importance for {modality} modality: {outcome}, {fusion_type}."
        )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"SHAP summary plot for {modality} modality saved to {save_path}.")


def aggregate_ts(data):
    """
    Aggregate timeseries SHAP values using mean pooling, ignoring missing values.

    Args:
        data (list or np.ndarray): List of timeseries SHAP arrays.

    Returns:
        np.ndarray: Aggregated SHAP values.
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
    Generate and save SHAP local-level decision plots for each modality and text highlight plot for the note segments.

    Args:
        shap_obj: List of SHAP explanation objects.
        risk_quantile: Risk quantile for plot title.
        figsize (tuple): Figure size.
        save_static_path (str): Path to save static modality plot.
        save_ts0_path (str): Path to save timeseries vitals plot.
        save_ts1_path (str): Path to save timeseries labs plot.
        save_nt_path (str): Path to save notes plot.
        shap_range (tuple): SHAP value range for plots.
        mm_scores (list): Multimodal SHAP scores.

    Returns:
        None
    """
    shap.initjs()
    ## Edit order of modalities if needed
    ts_order = [1, 2]
    notes_order = 3
    # Set format for SHAP values
    for i in range(len(shap_obj)):
        # shap_obj[i].values = shap_obj[i].values.round(6)
        shap_obj[i].base_values = shap_obj[i].base_values.round(3)
        if i in [1, 2]:
            shap_obj[i].data = shap_obj[i].data.round(2)
        # print(max(shap_obj[i][0].values.round(3)), min(shap_obj[i][0].values.round(3)))
        ### Static EHR modality
        if i == 0:
            plt.figure(figsize=figsize)
            shap_obj[i].data = np.where(shap_obj[i].data == 0, "No", shap_obj[i].data)
            shap_obj[i].data = np.where(
                shap_obj[i].data == "1", "Yes", shap_obj[i].data
            )
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
            plt.title(f"Static modality (RQ={risk_quantile}, TB-SHAP={mm_scores[0]}%).")
            plt.tight_layout()
            plt.savefig(save_static_path, dpi=300)
            plt.close()
        ### Timeseries modalities
        if i in ts_order:
            shap_obj[i].data = np.where(
                shap_obj[i].data == -1, "Missing", shap_obj[i].data
            )
            plt.figure(figsize=figsize)
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
                plt.title(
                    f"TS Vitals modality (RQ={risk_quantile}, TS-SHAP={mm_scores[1]}%)."
                )
            else:
                plt.title(
                    f"TS Labs modality (RQ={risk_quantile}, TS-SHAP={mm_scores[1]}%)."
                )

            plt.tight_layout()
            if i == 1:
                plt.savefig(save_ts0_path, bbox_inches="tight", dpi=300)
            else:
                plt.savefig(save_ts1_path, bbox_inches="tight", dpi=300)
            plt.close()
        ### Notes modality
        if i == notes_order:
            shap_obj[i].values = np.array([shap_obj[i].values])[0]
            shap_obj[i].data = shap_obj[i].data.astype("O")
            plot_highlighted_text_with_colorbar(
                shap_obj[i].values.round(3),
                shap_obj[i].data,
                shap_obj[i].base_values,
                mm_scores[2],
                save_path=save_nt_path,
                shap_range=shap_range,
            )

    print("SHAP local-level decision plots saved to disk.")


def _draw_token(ax, text, x, y, color, max_x, line_height):
    """
    Draw a single sentence token from a discharge sumamry with background color on the plot.

    Args:
        ax: Matplotlib axis.
        text (str): Token text.
        x (float): X position.
        y (float): Y position.
        color: Background color.
        max_x (float): Maximum X position.
        line_height (float): Height of a line.

    Returns:
        tuple: Updated (x, y) positions.
    """
    text_width = 0.01 * len(text)
    if x + text_width > max_x:
        x = 0.01
        y -= line_height
    ax.text(
        x,
        y,
        text,
        fontsize=10,
        va="top",
        ha="left",
        bbox=dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.1"),
    )
    x += text_width
    return x, y


def _draw_next_note(ax, x, y, line_height, note_text):
    """
    Draw a separator for the next note in the plot.

    Args:
        ax: Matplotlib axis.
        x (float): X position.
        y (float): Y position.
        line_height (float): Height of a line.
        note_text (str): Separator text.

    Returns:
        tuple: Updated (x, y) positions.
    """
    x = 0.01
    y -= line_height
    ax.text(x, y, note_text, fontsize=11, va="top", ha="left", color="black")
    y -= line_height
    x = 0.01
    return x, y


def _process_token_line(ax, token_line, val, cmap, norm, x, y, max_x, line_height):
    """
    Process and draw a line of tokens, handling note separators.

    Args:
        ax: Matplotlib axis.
        token_line (str): Line of tokens.
        val (float): SHAP value.
        cmap: Colormap.
        norm: Normalization for colormap.
        x (float): X position.
        y (float): Y position.
        max_x (float): Maximum X position.
        line_height (float): Height of a line.

    Returns:
        tuple: Updated (x, y, token_line).
    """
    if "<ENDNOTE> <STARTNOTE>" in token_line:
        parts = token_line.split("<ENDNOTE> <STARTNOTE>")
        for idx, part in enumerate(parts):
            if part.strip():
                color = cmap(norm(val))
                x, y = _draw_token(ax, part, x, y, color, max_x, line_height)
            if idx < len(parts) - 1:
                x, y = _draw_next_note(
                    ax, x, y, line_height, "------------NEXT NOTE-------------"
                )
        token_line = ""
    else:
        color = cmap(norm(val))
        x, y = _draw_token(ax, token_line, x, y, color, max_x, line_height)
        x = 0.01
        y -= line_height
        token_line = ""
    return x, y, token_line


def _render_highlighted_text(
    ax, shap_values, text_tokens, norm, cmap, line_height=0.08, max_x=0.98
):
    """
    Render highlighted text with color background for SHAP values.

    Args:
        ax: Matplotlib axis.
        shap_values (np.ndarray): SHAP values for each token.
        text_tokens (list): List of text tokens.
        norm: Normalization for colormap.
        cmap: Colormap.
        line_height (float): Height of a line.
        max_x (float): Maximum X position.

    Returns:
        float: Final Y position after rendering.
    """
    x = 0.01
    y = 0.95
    for token, val in zip(text_tokens, shap_values, strict=False):
        words = token.split()
        word_count = 0
        token_line = ""
        for word in words:
            token_line += word + " "
            word_count += 1
            # New line every 15 words
            if word_count % 15 == 0:
                x, y, token_line = _process_token_line(
                    ax, token_line, val, cmap, norm, x, y, max_x, line_height
                )
            # New line after every '#' symbol in word
            if "#" in word:
                color = cmap(norm(val))
                x, y = _draw_token(ax, token_line, x, y, color, max_x, line_height)
                x = 0.01
                y -= line_height
                token_line = ""
        if token_line.strip():
            if "<ENDNOTE> <STARTNOTE>" in token_line:
                parts = token_line.split("<ENDNOTE> <STARTNOTE>")
                for idx, part in enumerate(parts):
                    if part.strip():
                        color = cmap(norm(val))
                        x, y = _draw_token(
                            ax, part + " ", x, y, color, max_x, line_height
                        )
                    if idx < len(parts) - 1:
                        x, y = _draw_next_note(
                            ax,
                            x,
                            y,
                            line_height,
                            "------------NEXT NOTE---------------",
                        )
            else:
                color = cmap(norm(val))
                x, y = _draw_token(
                    ax, token_line + " ", x, y, color, max_x, line_height
                )
        else:
            x = 0.01
    return y


def plot_highlighted_text_with_colorbar(
    shap_values,
    text_tokens,
    expected_value,
    mm_score,
    figsize=(10, 4),
    cmap="coolwarm",
    save_path=None,
    shap_range=None,
):
    """
    Display a highlighted text plot and a colorbar based on SHAP values.

    Args:
        shap_values (np.ndarray): SHAP values for each token (1D array).
        text_tokens (list of str): List of text tokens (words or sentences).
        expected_value (float): Reference value for colorbar label.
        mm_score (float): MM-SHAP dependence score for notes modality.
        figsize (tuple): Figure size.
        cmap (str): Matplotlib colormap.
        save_path (str): If provided, saves the plot to this path.
        shap_range (tuple): Min/Max SHAP values for colorbar.

    Returns:
        None
    """
    norm = mpl.colors.Normalize(vmin=shap_range[0], vmax=shap_range[1])
    cmap = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Render highlighted text using helper
    y = _render_highlighted_text(ax, shap_values, text_tokens, norm, cmap)

    # Draw a bounding box around the text area
    min_y = y - 0.08
    rect = mpl.patches.FancyBboxPatch(
        (0, min_y),
        1,
        0.98 - min_y,
        boxstyle="round,pad=0.01",
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(rect)

    # Add colorbar at the top and center it
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=0.15,
        pad=0.02,
        anchor=(0.5, 1.0),
        location="top",
    )
    cbar.set_label(
        f"SHAP Values (Ref={expected_value:.3f}, TX-SHAP={mm_score}%)",
        fontsize=14,
        labelpad=10,
    )
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
