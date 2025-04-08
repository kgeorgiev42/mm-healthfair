import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_feature_names(test_set, modality_type='tabular'):
    """
    Get feature names based on the modality type.

    Parameters:
    - test_set: The test set object containing feature information.
    - modality_type: Type of data ('tabular', 'ts-vitals', 'ts-labs' or 'notes').

    Returns:
    - feature_names: List of feature names.
    """
    if modality_type == 'tabular':
        feature_names = test_set.get_feature_list()
    elif modality_type == 'ts-vitals':
        feature_names = test_set.get_feature_list(f"dynamic_1")
    elif modality_type == 'ts-labs':
        feature_names = test_set.get_feature_list(f"dynamic_0")
    elif modality_type == 'notes':
        feature_names = test_set.get_feature_list(f"notes")['sentence']
    return feature_names

def get_shap_values(model, batch,
                    modality_type='tabular'):
    """
    Calculate SHAP values for a given model and data batch while inferring feature names.

    Parameters:
    - model: The trained model for which to calculate SHAP values.
    - X: The input data (features) for which to calculate SHAP values.
    - feature_names: List of feature names (optional).
    - modality_type: Type of data ('tabular', 'ts-vitals', 'ts-labs' or 'notes').

    Returns:
    - shap_values: The calculated SHAP values.
    """
    if modality_type == 'tabular':
        x_test = batch[0]
        emb = model.embed_static(x_test)
    elif modality_type == 'ts-vitals':
        x_test = batch[2][1]
        emb = model.embed_timeseries[1](x_test)
    elif modality_type == 'ts-labs':
        x_test = batch[2][0]
        emb = model.embed_timeseries[0](x_test)
    elif modality_type == 'notes':
        x_test = batch[4]
        emb = model.embed_notes(x_test)
    # Use DeepExplainer for other models
    explainer = shap.DeepExplainer(emb, x_test)
    if modality_type == 'tabular':
        shap_values = explainer.shap_values(x_test)
    else:
        shap_values = explainer.shap_values(x_test, check_additivity=False)

    return shap_values, x_test

def get_shap_summary_plot(shap_values, features, feature_names, outcome='In-hospital Death',
                          fusion_type=None, modality=None, max_features=20,
                          figsize=(8, 5), save_path=None):
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
    shap.summary_plot(shap_values, features, feature_names=feature_names, figsize=figsize,
                      plot_type='violin', max_display=max_features, figsize=(8, 5), show=False)
    plt.title(f'SHAP Global Importance: {outcome}, {fusion_type}({modality}).')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to {save_path}")

def get_shap_partial_dependence(shap_values, feature_index,
                                attr_group, attr_disp,
                            outcome='In-hospital Death', fusion_type=None, 
                            modalities=None,
                            save_dir=None, figsize=(8, 5), verbose=True):
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
        color=shap_values[:, feature_index], figsize=figsize, show=False
    )
    plt.title(f'SHAP Partial Dependence By Attribute {attr_disp}: {outcome}, {fusion_type}({modalities}).')
    plt.savefig(os.path.join(save_dir, f'{outcome}_pdp_{feature_index}_by_{attr_group}.png'), bbox_inches='tight', dpi=300)
    plt.close()
    if verbose:
        print(f"Partial dependence plot saved to {save_dir + f'{outcome}_pdp_{feature_index}_by_{attr_group}.png'}")

def get_shap_group_difference(shap_values, risk_dict, risk_quantile: int = 10, feature_names=None,
                            outcome='In-hospital Death', fusion_type=None, 
                            modalities=None, max_features=20,
                            save_path=None, figsize=(8, 5), verbose=True):
    """
    Generate a SHAP group difference plot.
    """
    if verbose:
        print(f"Generating SHAP group difference plot for risk quantile {risk_quantile}...")

    risk_quantiles = risk_dict['risk_quantile']
    risk_mask = risk_quantiles == risk_quantile
    plt.figure()
    shap.plots.group_difference(
        shap_values, group_mask=risk_mask, feature_names=feature_names,
        max_display=max_features, figsize=figsize, show=False
    )
    plt.title(f'SHAP Group Difference By Risk Quantile {risk_quantile}: {outcome}, {fusion_type}({modalities}).')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    if verbose:
        print(f"Group difference plot saved to {save_path}")

def get_standard_force_plot(explainer, shap_idx, feature_idx, y_test, y_hat, prob,
                        outcome='In-hospital Death', fusion_type=None, 
                        subject_idx: int=0, risk_quantile: int=10,
                        modality='static', feature_names=None,
                        save_path=None, figsize=(8, 5), verbose=True):
    """
    Generate a SHAP force plot.
    """
    if verbose:
        print(f"Generating Force plot ({modality}) for Subject {subject_idx} in Risk Quantile {risk_quantile}...")
    
    y_test = 'Y' if y_test else 'N'
    y_hat = 'Y' if y_hat else 'N'
    plt.figure()
    shap.plots.force(
        explainer.expected_value, shap_idx, feature_idx, feature_names=feature_names,
        matplotlib=True, figsize=figsize, show=False
    )
    plt.title(f'SHAP Force Plot For Subject {subject_idx} in Risk Quantile {risk_quantile}: {outcome}, {fusion_type}({modality}).')
    plt.suptitle(f"Truth: {y_test}, Predict: {y_hat}, Prob: {round(prob, 2)*100}%")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    if verbose:
        print(f"Force plot saved to {save_path}")

def get_shap_text_plot(explainer, shap_values, text_data, y_test, y_hat, prob,
                       outcome='In-hospital Death', fusion_type=None, 
                       subject_idx: int=0, risk_quantile: int=10,
                       save_path=None, verbose=True):
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
        print(f"Generating SHAP text plot for Subject {subject_idx} in Risk Quantile {risk_quantile}...")
    y_test = 'Y' if y_test else 'N'
    y_hat = 'Y' if y_hat else 'N'
    # Generate the SHAP text plot
    plt.figure()
    shap.plots.text(
        explainer.expected_value, shap_values, text_data, show=False
    )
    plt.title(f'SHAP Text Plot For Subject {subject_idx} in Risk Quantile {risk_quantile}: {outcome}, {fusion_type}(notes).')
    plt.suptitle(f"Truth: {y_test}, Predict: {y_hat}, Prob: {round(prob, 2)*100}%")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    if verbose:
        print(f"Text plot saved to {save_path}")