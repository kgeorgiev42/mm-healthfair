import os

import matplotlib.pyplot as plt
import shap

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
            embed = embed.view(embed.size(0), embed.size(1), -1)  # Flatten the final dimension
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
        #print("Computing SHAP values for static modality...")
        wrapper_static = ModelWrapper(model, "static", total_dim=64, target_size=1).to(device)
        explainer_static = shap.DeepExplainer(wrapper_static, s.to(device))
        shap_values["static"] = explainer_static.shap_values(s.to(device), check_additivity=False)
    if "timeseries" in modalities:
        #print("Computing SHAP values for timeseries modality...")
        ts_shap_values = {}
        for i in range(num_ts):
            wrapper_ts = ModelWrapper(model, "timeseries", total_dim=128, target_size=1, ts_ind=i).to(device)
            explainer_ts = shap.DeepExplainer(wrapper_ts, ts_data['dynamic' + str(i)].to(device))
            ts_shap_values['dynamic' + str(i)] = explainer_ts.shap_values(ts_data['dynamic' + str(i)].to(device), check_additivity=False)
        shap_values["timeseries"] = ts_shap_values
    if "notes" in modalities:
        #print("Computing SHAP values for notes modality...")
        wrapper_notes = ModelWrapper(model, "notes", total_dim=64, target_size=1).to(device)
        explainer_notes = shap.DeepExplainer(wrapper_notes, n.to(device))
        shap_values["notes"] = explainer_notes.shap_values(n.to(device), check_additivity=False)

    return shap_values


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
    plt.grid('both')
    if heatmap:
        plt.title(f"Heatmap view for {modality} modality.")
    else:
        plt.title(f"SHAP Global Importance for {modality} modality: {outcome}, {fusion_type}.")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"SHAP summary plot for {modality} modality saved to {save_path}.")


def get_shap_partial_dependence(
    shap_values,
    feature_index,
    attr_group,
    attr_disp,
    outcome="In-hospital Death",
    fusion_type=None,
    modalities=None,
    save_dir=None,
    figsize=(8, 5),
    verbose=True,
):
    """
    Generate a SHAP partial dependence plot.

    Parameters:
    - shap_values: The calculated SHAP values.
    - features: The input features for which to calculate SHAP values.
    - feature_index: Index of the feature to plot.
    - title: Title for the plot (optional).

    Returns:
    - None: Displays the SHAP partial dependence plot.
    """
    if verbose:
        print(f"Generating SHAP PDP between {attr_group} and {feature_index}...")
    plt.figure()
    shap.plots.scatter(
        shap_values[:, attr_group],
        color=shap_values[:, feature_index],
        figsize=figsize,
        show=False,
    )
    plt.title(
        f"SHAP Partial Dependence By Attribute {attr_disp}: {outcome}, {fusion_type}({modalities})."
    )
    plt.savefig(
        os.path.join(save_dir, f"{outcome}_pdp_{feature_index}_by_{attr_group}.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    if verbose:
        print(
            f"Partial dependence plot saved to {save_dir + f'{outcome}_pdp_{feature_index}_by_{attr_group}.png'}"
        )


def get_shap_group_difference(
    shap_values,
    risk_dict,
    risk_quantile: int = 10,
    feature_names=None,
    outcome="In-hospital Death",
    fusion_type=None,
    modalities=None,
    max_features=20,
    save_path=None,
    figsize=(8, 5),
    verbose=True,
):
    """
    Generate a SHAP group difference plot.
    """
    if verbose:
        print(
            f"Generating SHAP group difference plot for risk quantile {risk_quantile}..."
        )

    risk_quantiles = risk_dict["risk_quantile"]
    risk_mask = risk_quantiles == risk_quantile
    plt.figure()
    shap.plots.group_difference(
        shap_values,
        group_mask=risk_mask,
        feature_names=feature_names,
        max_display=max_features,
        figsize=figsize,
        show=False,
    )
    plt.title(
        f"SHAP Group Difference By Risk Quantile {risk_quantile}: {outcome}, {fusion_type}({modalities})."
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    if verbose:
        print(f"Group difference plot saved to {save_path}")


def get_standard_force_plot(
    explainer,
    shap_idx,
    feature_idx,
    y_test,
    y_hat,
    prob,
    outcome="In-hospital Death",
    fusion_type=None,
    subject_idx: int = 0,
    risk_quantile: int = 10,
    modality="static",
    feature_names=None,
    save_path=None,
    figsize=(8, 5),
    verbose=True,
):
    """
    Generate a SHAP force plot.
    """
    if verbose:
        print(
            f"Generating Force plot ({modality}) for Subject {subject_idx} in Risk Quantile {risk_quantile}..."
        )

    y_test = "Y" if y_test else "N"
    y_hat = "Y" if y_hat else "N"
    plt.figure()
    shap.plots.force(
        explainer.expected_value,
        shap_idx,
        feature_idx,
        feature_names=feature_names,
        matplotlib=True,
        figsize=figsize,
        show=False,
    )
    plt.title(
        f"SHAP Force Plot For Subject {subject_idx} in Risk Quantile {risk_quantile}: {outcome}, {fusion_type}({modality})."
    )
    plt.suptitle(f"Truth: {y_test}, Predict: {y_hat}, Prob: {round(prob, 2)*100}%")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()
    if verbose:
        print(f"Force plot saved to {save_path}")


def get_shap_text_plot(
    explainer,
    shap_values,
    text_data,
    y_test,
    y_hat,
    prob,
    outcome="In-hospital Death",
    fusion_type=None,
    subject_idx: int = 0,
    risk_quantile: int = 10,
    save_path=None,
    verbose=True,
):
    """
    Generate a SHAP text plot for the notes modality.

    Parameters:
    - explainer: The SHAP explainer object.
    - shap_values: The calculated SHAP values for the text data.
    - text_data: The input text data for which SHAP values are calculated.
    - outcome: The outcome being explained (default: 'In-hospital Death').
    - fusion_type: The fusion method used in the model (default: None).
    - subject_idx: The index of the subject being explained.
    - risk_quantile: The risk quantile of the subject.
    - save_path: Path to save the SHAP text plot.
    - verbose: Whether to print additional information (default: True).

    Returns:
    - None: Saves the SHAP text plot to the specified path.
    """
    if verbose:
        print(
            f"Generating SHAP text plot for Subject {subject_idx} in Risk Quantile {risk_quantile}..."
        )
    y_test = "Y" if y_test else "N"
    y_hat = "Y" if y_hat else "N"
    # Generate the SHAP text plot
    plt.figure()
    shap.plots.text(explainer.expected_value, shap_values, text_data, show=False)
    plt.title(
        f"SHAP Text Plot For Subject {subject_idx} in Risk Quantile {risk_quantile}: {outcome}, {fusion_type}(notes)."
    )
    plt.suptitle(f"Truth: {y_test}, Predict: {y_hat}, Prob: {round(prob, 2)*100}%")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    if verbose:
        print(f"Text plot saved to {save_path}")
