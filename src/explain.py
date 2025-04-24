import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import shap
import toml
import torch
from datasets import CollateFn, CollateTimeSeries, MIMIC4Dataset
from models import MMModel
from torch import concat
from torch.utils.data import DataLoader
from utils.functions import load_pickle
from utils.shap_utils import (
    get_feature_names,
    get_shap_group_difference,
    get_shap_partial_dependence,
    get_shap_summary_plot,
    get_shap_text_plot,
    get_shap_values,
    get_standard_force_plot,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal SHAP feature importance analysis."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the pickled data dictionary generated from prepare_data.py.",
        default="../outputs/processed_data/mmfair_feat.pkl",
    )
    parser.add_argument(
        "--col_path",
        "-p",
        type=str,
        help="Path to the pickled column dictionary generated from prepare_data.py.",
        default="../outputs/processed_data/mmfair_cols.pkl",
    )
    parser.add_argument(
        "--ids_path",
        "-i",
        type=str,
        help="Directory containing test ids.",
        default="../outputs/processed_data",
    )
    parser.add_argument(
        "--exp_path",
        "-x",
        type=str,
        help="Directory to store explanation plots.",
        default="../outputs/explanations",
    )
    parser.add_argument(
        "--eval_path",
        "-e",
        type=str,
        help="Evaluation path for obtaining risk dictionary as .pkl file (eval_path+model_path).",
        default="../outputs/evaluation",
    )
    parser.add_argument(
        "--outcome",
        "-o",
        type=str,
        help="Binary outcome to use for multimodal learning (one of the labels in targets.toml)."
        "Defaults to prediction of in-hospital death.",
        default="in_hosp_death",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing the saved model metadata.",
        default="logs/nhs-mm-healthfair",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="Directory pointing to the saved model checkpoint (must be inside model_dir).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="../config/model.toml",
        help="Path to config toml file containing model training parameters.",
    )
    parser.add_argument(
        "--targets",
        "-t",
        type=str,
        default="../config/targets.toml",
        help="Path to config toml file containing lookup fields and outcomes.",
    )
    parser.add_argument(
        "--exp_mode",
        type=str,
        default="global",
        help="Use global or local explanations. If global, uses DeepExplainer over static and timeseries modalities in the test set."
        "If local, uses DeepExplainer to explain individual predictions over all modalities (requires a fused static-timeseries-text model).",
    )
    parser.add_argument(
        "--group_by_risk",
        action="store_true",
        help="Show stratified risk difference plots by attribute if using exp_mode=global.",
    )
    parser.add_argument(
        "--diff_risk_quantile",
        "-dq",
        type=int,
        default=10,
        help="Risk quantile to use for global risk difference explanations if group_by_risk=True (Default is 10, treated as highest-risk for deciles).",
    )
    parser.add_argument(
        "--diff_modality",
        type=str,
        default="static",
        help="Modality to use for risk difference plots if group_by_risk=True (defaults to static).",
    )
    parser.add_argument(
        "--pdp_analysis",
        action="store_true",
        help="Generate partial dependence plots (PDP) for a feature of interest if using exp_mode=global.",
    )
    parser.add_argument(
        "--pdp_feature",
        type=str,
        default="anchor_age",
        help="Feature to use for partial dependence plots (PDP), looking at impact by sensitive attribute, e.g. one of top ranking features in SHAP summary plot.",
    )
    parser.add_argument(
        "--global_max_features",
        "-g",
        type=int,
        default=20,
        help="Top N features to plot in global-level explanations.",
    )
    parser.add_argument(
        "--local_samples",
        "-s",
        type=int,
        default=10,
        help="# subjects to generate local-level explanations for if using exp_model=local.",
    )
    parser.add_argument(
        "--local_risk_group",
        "-r",
        type=int,
        default=10,
        help="Risk quantile to use for local-level explanations (generated in evaluate.py).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        default=False,
        help="Control verbosity.",
    )
    args = parser.parse_args()
    config = toml.load(args.config)
    targets = toml.load(args.targets)
    ### General setup
    outcomes = targets["outcomes"]["labels"]
    outcomes_disp = targets["outcomes"]["display"]
    outcomes_col = targets["outcomes"]["colormap"]
    attributes = targets["attributes"]["labels"]
    attr_disp = targets["attributes"]["display"]
    batch_size = config["data"]["batch_size"]
    ### Model-specific setup
    model_path = os.path.join(
        os.path.join(args.model_dir, args.model_path), args.model_path + ".ckpt"
    )
    risk_path = os.path.join(
        os.path.join(args.eval_path, args.model_path), "pf_" + args.model_path + ".pkl"
    )
    exp_path = os.path.join(args.exp_path, args.model_path)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    num_workers = config["data"]["num_workers"]
    ### Infer fusion method and modalities from model_path
    modalities = []
    for arg in args.model_path.split("_"):
        if "static" in arg:
            modalities.append("static")
        elif "timeseries" in arg:
            modalities.append("timeseries")
        elif "notes" in arg:
            modalities.append("notes")
    static_only = True if (len(modalities) == 1) and ("static" in modalities) else False
    with_notes = True if "notes" in modalities else False
    fusion_method = "None"
    if "mag" in args.model_path:
        fusion_method = "EF-mag"
    elif "concat" in args.model_path:
        fusion_method = "EF-concat"
    if args.outcome not in outcomes:
        print(f"Outcome {args.outcome} must be included in targets.toml.")
        sys.exit()
    outcome_idx = outcomes.index(args.outcome)
    ### Set plotting style
    plt.rcParams.update(
        {"font.size": 13, "font.weight": "normal", "font.family": "serif"}
    )
    print("------------------------------------------")
    print("MMHealthFair: Multimodal SHAP feature importance analysis")
    print("------------------------------------------")
    print(f'Evaluating feature importances for outcome "{outcomes_disp[outcome_idx]}"')
    print(f"Modalities used: {modalities}")
    print(f"Fusion method: {fusion_method}")
    print(f"Mode: {args.exp_mode}")
    if args.exp_mode == "global":
        print(f"Global max features: {args.global_max_features}")
        print(f"Group by risk: {args.group_by_risk}")
        print(f"Risk difference quantile: {args.diff_risk_quantile}")
        print(f"PDP analysis: {args.pdp_analysis}")
    else:
        print(f"Local samples: {args.local_samples}")
        print(f"Risk group target: {args.local_risk_group}")
    print("------------------------------------------")
    # Get test ids
    if (
        len(
            glob.glob(
                os.path.join(args.ids_path, "testing_ids_" + args.outcome + ".csv")
            )
        )
        == 0
    ):
        print(f"No test ids found for outcome {args.outcome}. Exiting..")
        sys.exit()

    if (len(model_path) == 0) and (not args.group_models):
        print(f"No model found at {args.model_path}. Exiting..")
        sys.exit()

    if not os.path.exists(risk_path):
        print(f"No risk dictionary found at {risk_path}. Exiting..")
        sys.exit()

    test_ids = (
        pl.read_csv(os.path.join(args.ids_path, "testing_ids_" + args.outcome + ".csv"))
        .select("subject_id")
        .to_numpy()
        .flatten()
    )
    risk_dict = load_pickle(risk_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = MIMIC4Dataset(
        args.data_path,
        args.col_path,
        "test",
        ids=test_ids,
        static_only=static_only,
        with_notes=with_notes,
    )
    test_set.print_label_dist()
    ### Load model
    model = MMModel.load_from_checkpoint(checkpoint_path=model_path)
    dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateFn() if static_only else CollateTimeSeries(),
        persistent_workers=True if num_workers > 0 else False,
    )
    batch = next(iter(dataloader))
    # trainer = Trainer(accelerator="auto")
    # output = trainer.predict(model, dataloaders=dataloader)
    # prob = np.array(concat([out[0] for out in output])).flatten().astype(np.float32)
    ### SHAP setup
    ### Collect feature explanations
    if args.exp_mode == "global":
        shap_dict = {}
        shap_ts = {}
        shap_notes = {}
        shap_ehr = {}
        for modality in modalities:
            if modality == "static":
                print("Processing explanations for static EHR modality...")
                feature_names = get_feature_names(test_set, modality_type="tabular")
                shap_scores = get_shap_values(model, dataloader, device=device)
                print(shap_scores.shape, shap_scores[0])
                # Reshape shap scores to batch_size x length of feature_names
                # shap_scores = shap_scores.reshape(shap_scores.shape[0], len(feature_names))
                get_shap_summary_plot(
                    shap_scores,
                    feature_names,
                    outcome=outcomes_disp[outcome_idx],
                    fusion_type=fusion_method,
                    modality=modality,
                    max_features=args.global_max_features,
                    figsize=(8, 5),
                    save_path=os.path.join(exp_path, f"shap_summary_{modality}.png"),
                )
                shap_dict.update(
                    {
                        feature: shap_scores[:, idx]
                        for idx, feature in enumerate(feature_names)
                    }
                )
                shap_ehr.update(
                    {
                        feature: shap_scores[:, idx]
                        for idx, feature in enumerate(feature_names)
                    }
                )
            elif modality == "timeseries":
                print("Processing explanations for timeseries modality...")
                features_vitals = get_feature_names(test_set, modality_type="ts-vitals")
                features_labs = get_feature_names(test_set, modality_type="ts-labs")
                feature_names = features_vitals + features_labs
                # shap_concat = concat([shap_scores_vitals, shap_scores_labs], dim=1)
                # feat_concat = concat([x_test_vit, x_test_lab], dim=1)
                ### If using timeseries, plot summary over a single timepoint (t=0)
                shap_scores = shap_scores.mean(axis=3)[:, 0, :]
                get_shap_summary_plot(
                    shap_scores,
                    feature_names,
                    outcome=outcomes_disp[outcome_idx],
                    fusion_type=fusion_method,
                    modality=modality,
                    max_features=args.global_max_features,
                    figsize=(8, 5),
                    save_path=os.path.join(exp_path, f"shap_summary_{modality}.png"),
                )
                shap_dict.update(
                    {
                        feature: shap_scores[:, idx]
                        for idx, feature in enumerate(feature_names)
                    }
                )
                shap_ts.update(
                    {
                        feature: shap_scores[:, idx]
                        for idx, feature in enumerate(feature_names)
                    }
                )
            elif modality == "notes":
                print("Processing explanations for notes modality...")
                feature_names = get_feature_names(test_set, modality_type="notes")
                shap_scores, x_test = get_shap_values(
                    model, batch, test_set, modality_type="notes"
                )
                get_shap_summary_plot(
                    shap_scores,
                    x_test,
                    feature_names,
                    outcome=outcomes_disp[outcome_idx],
                    fusion_type=fusion_method,
                    modality=modality,
                    max_features=args.global_max_features,
                    figsize=(8, 5),
                    save_path=os.path.join(exp_path, f"shap_summary_{modality}.png"),
                )
                shap_dict.update(
                    {
                        feature: shap_scores[:, idx]
                        for idx, feature in enumerate(feature_names)
                    }
                )
                shap_notes.update(
                    {
                        feature: shap_scores[:, idx]
                        for idx, feature in enumerate(feature_names)
                    }
                )
        ### Plot PDP for a feature of interest
        if args.pdp_analysis:
            if args.pdp_feature not in shap_dict.keys():
                print(
                    f"Feature {args.pdp_feature} not found in SHAP summary plot. Exiting..."
                )
                sys.exit()
            print(f"Generating PDP for feature {args.pdp_feature}...")
            for attribute, disp in zip(attributes, attr_disp, strict=False):
                get_shap_partial_dependence(
                    shap_dict,
                    args.pdp_feature,
                    attribute,
                    disp,
                    outcome=outcomes_disp[outcome_idx],
                    fusion_type=fusion_method,
                    modalities=modalities,
                    save_dir=exp_path,
                    verbose=args.verbose,
                )

        ### Plot stratified risk difference plots by attribute
        if args.global_group_by_risk:
            print(
                f"Generating risk difference plots (one vs rest) with target quantile {args.diff_risk_quantile}..."
            )
            if args.diff_modality == "static":
                shap_diff = shap_ehr
            elif args.diff_modality == "timeseries":
                shap_diff = shap_ts
            else:
                shap_diff = shap_notes

            get_shap_group_difference(
                shap_diff,
                risk_dict,
                args.diff_risk_quantile,
                feature_names=None,
                outcome=outcomes_disp[outcome_idx],
                fusion_type=fusion_method,
                modalities=modalities,
                max_features=args.global_max_features,
                save_path=os.path.join(
                    exp_path,
                    f"shap_gdiff_{args.diff_modality}_rg_{args.diff_risk_quantile}_vs_rest.png",
                ),
                figsize=(8, 5),
                verbose=True,
            )

    if args.exp_mode == "local":
        print(
            f"Generating local-level SHAP explanations for {args.local_samples} subjects at risk quantile {args.local_risk_group}..."
        )
        # Get N random subjects in the specified risk group
        risk_idx = pd.DataFrame(risk_dict)
        risk_idx = (
            risk_idx[risk_dict["risk_quantile"] == args.local_risk_group]
            .sample(args.local_samples, random_state=0)
            .index.tolist()
        )
        for i, subject in enumerate(risk_idx):
            ## Get local-level predictions
            # y_test_subj = y_test[subject]
            # y_hat_subj = y_hat[subject]
            # prob_subj = prob[subject]
            ## Get SHAP values for each modality
            for modality in modalities:
                if modality == "static":
                    feature_names = test_set.get_feature_list()
                    x_test = batch[0][subject]
                    emb = model.embed_static(x_test).cpu().numpy()
                    explainer = shap.DeepExplainer(emb, x_test)
                    shap_values = explainer.shap_values(x_test)
                    get_standard_force_plot(
                        explainer,
                        shap_values,
                        x_test,
                        y_test=None,
                        y_hat=None,
                        prob=None,
                        outcome=outcomes_disp[outcome_idx],
                        fusion_type=fusion_method,
                        subject_idx=i + 1,
                        risk_quantile=args.local_risk_group,
                        feature_names=feature_names,
                        save_path=os.path.join(
                            exp_path,
                            f"shap_local_{modality}_rg_{args.local_risk_group}_subj_{i+1}.png",
                        ),
                        verbose=args.verbose,
                    )
                elif modality == "timeseries":
                    feature_names = test_set.get_feature_list()
                    # x_test_vit = batch[2][1][subject]
                    # x_test_lab = batch[2][0][subject]
                    emb_vit = model.embed_timeseries[1](x_test)
                    emb_lab = model.embed_timeseries[0](x_test)
                    emb = concat([emb_vit, emb_lab], dim=1).cpu().numpy()
                    # x_test = concat([x_test_vit, x_test_lab], dim=1).cpu().numpy()
                    ### If using timeseries, plot summary over a single timepoint (t=0)
                    # x_test = feat_concat[:, 0, :]
                    explainer = shap.DeepExplainer(emb, x_test)
                    shap_values = explainer.shap_values(x_test)
                    get_standard_force_plot(
                        explainer,
                        shap_values,
                        x_test,
                        y_test=None,
                        y_hat=None,
                        prob=None,
                        outcome=outcomes_disp[outcome_idx],
                        fusion_type=fusion_method,
                        subject_idx=i + 1,
                        risk_quantile=args.local_risk_group,
                        feature_names=feature_names,
                        save_path=os.path.join(
                            exp_path,
                            f"shap_local_{modality}_rg_{args.local_risk_group}_subj_{i+1}.png",
                        ),
                        verbose=args.verbose,
                    )

                elif modality == "notes":
                    feature_names = get_feature_names(test_set, modality_type="notes")
                    # x_test = batch[4]
                    emb = model.embed_notes(x_test).cpu().numpy()
                    explainer = shap.DeepExplainer(emb, x_test)
                    # y_test_subj = y_test[subject]
                    # y_hat_subj = y_hat[subject]
                    # prob_subj = prob[subject]
                    # Generate SHAP text plot
                    get_shap_text_plot(
                        explainer=explainer,
                        shap_values=shap_scores,
                        text_data=x_test,
                        y_test=None,
                        y_hat=None,
                        prob=None,
                        outcome=outcomes_disp[outcome_idx],
                        fusion_type=fusion_method,
                        subject_idx=1,  # Example subject index
                        risk_quantile=args.local_risk_group,
                        save_path=os.path.join(
                            exp_path,
                            f"shap_local_{modality}_rg_{args.local_risk_group}_subj_{i+1}.png",
                        ),
                        verbose=args.verbose,
                    )

        # Get model predictions
        # y_test = batch[0][:, outcome_idx].cpu().numpy()
        # y_hat = model(batch).cpu().numpy()
        # prob = model(batch, return_prob=True).cpu().numpy()
        # x_test = batch[1].cpu().numpy()
        # Get SHAP values
        # shap_values = get_shap_values(model, batch, test_set, modality_type="tabular")
