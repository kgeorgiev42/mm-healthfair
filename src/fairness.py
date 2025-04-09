import argparse
import os
import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import toml
from datasets import CollateFn, CollateTimeSeries, MIMIC4Dataset

from lightning.pytorch import Trainer

from models import MMModel
from torch import concat
from torch.utils.data import DataLoader
from utils.functions import load_pickle, save_pickle
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_ratio,
    equalized_odds_ratio,
    equal_opportunity_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from utils.fairness_utils import plot_bar_metric_frame, get_bootstrapped_fairness_measures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal fairness evaluator pipeline.")
    parser.add_argument(
        "eval_path",
        type=str,
        help="Evaluation path for obtaining risk dictionary as .pkl file (generated from evaluate.py)." \
        "Must include the directory name as (outcome_name)_(fusion_type)_(modalities), containing the .pkl file named as pf_(outcome_name)_(fusion_type)_(modalities).pkl.",
        default="../outputs/evaluation",
    )
    parser.add_argument(
        "--fair_path",
        "-f",
        type=str,
        help="Directory to store explanation plots.",
        default="../outputs/fairness",
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
        "--attr_path",
        "-a",
        type=str,
        help="Directory containing attributes metadata (original ehr_static.csv).",
        default="../outputs/sample_data/ehr_static.csv",
    )
    parser.add_argument(
        "--outcome",
        "-o",
        type=str,
        help="Binary outcome to use for multimodal evaluation (one of the labels in targets.toml)."
        "Defaults to prediction of in-hospital death.",
        default="in_hosp_death",
    )
    parser.add_argument(
        "--targets",
        "-t",
        type=str,
        default="../config/targets.toml",
        help="Path to config toml file containing lookup fields and outcomes.",
    )
    parser.add_argument(
        "--group_by_risk",
        action="store_true",
        help="If true, will generate a trajectory lineplot of the fairness measures across each risk group (as generated in evaluate.py).",
    )
    parser.add_argument(
        "--plot_grouped_bar",
        action="store_true",
        help="If true, will use the Fairlearn API to generate a grouped barplot of the fairness measures over each sensitive attribute (as specified in targets.toml).",
    )
    parser.add_argument(
        "--across_models",
        action="store_true",
        help="If true, will generate a trajectory lineplot of the fairness measures across multiple models (specified in paths within targets.toml).",
    )
    parser.add_argument(
        "--threshold_method",
        "-th",
        type=str,
        default="yd",
        help="Method to use for thresholding positive and negative classes under class imbalance. " \
        "Options are 'yd' (Youden's J statistic) or 'f1' (Maximum achievable F1-score).",
    )
    parser.add_argument(
        "--boot_samples",
        "-b",
        type=int,
        default=1000,
        help="Number of bootstrap samples to use for calculating confidence intervals.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for bootstrapping. Defaults to 42."
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
    targets = toml.load(args.targets)
    ### General setup
    outcomes = targets["outcomes"]["labels"]
    outcomes_disp = targets["outcomes"]["display"]
    outcomes_col = targets["outcomes"]["colormap"]
    attributes = targets["attributes"]["labels"]
    attr_disp = targets["attributes"]["display"]
    threshold_method = args.threshold_method
    ### Model-specific setup
    model_path = os.path.join(os.path.join(args.model_dir, args.model_path), args.model_path + ".ckpt")
    risk_path = os.path.join(os.path.join(args.eval_path, args.model_path), 'pf_' + args.model_path + ".pkl")
    fair_path = os.path.join(args.fair_path, args.model_path)
    if not os.path.exists(fair_path):
        os.makedirs(fair_path)
    ### Infer fusion method and modalities from model_path
    modalities = []
    for arg in args.model_path.split("_"):
        if 'static' in arg:
            modalities.append('static')
        elif 'timeseries' in arg:
            modalities.append('timeseries')
        elif 'notes' in arg:
            modalities.append('notes')
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
    print('------------------------------------------')
    print("MMHealthFair: Multimodal Fairness analysis")
    print('------------------------------------------')
    print(f'Evaluating fairness for outcome "{outcomes_disp[outcome_idx]}"')
    print(f'Modalities used: {modalities}')
    print(f'Fusion method: {fusion_method}')
    print(f'Across multiple models: {args.across_models}')
    ### Get test ids
    if (len(model_path) == 0) and (args.group_models == False):
        print(f"No model found at {args.model_path}. Exiting..")
        sys.exit()
    
    if not os.path.exists(risk_path):
        print(f"No risk dictionary found at {risk_path}. Exiting..")
        sys.exit()

    ### Get risk predictions
    risk_dict = load_pickle(risk_path)
    test_ids = risk_dict["test_ids"]
    y_test = risk_dict["y_test"]
    y_prob = risk_dict["y_prob"]
    ### Use Youden's J statistic of F1-max rather than sigmoid for best discrimination threshold as classes are imbalanced
    if threshold_method == "yd":
        thres = risk_dict["yd_idx"]
    elif threshold_method == "f1":
        thres = risk_dict["f1_thres"]
    y_hat = np.where(y_prob > thres, 1, 0)
    ### Get sensitive attributes data
    print("Reading attributes metadata for fairness analysis...")
    if not os.path.exists(args.attr_path):
        print("Attributes metadata not found. Exiting...")
        sys.exit()
    ehr_static = (pl.read_csv(args.attr_path)
                  .filter(pl.col("subject_id").is_in(list(map(int, test_ids))))
                  .select(attributes))
    
    # Get feature for all test_ids from metadata
    for pf, disp in zip(attributes, attr_disp):
        attr_pf = ehr_static.select(pl.col(pf))
        if args.verbose:
            print("---------------------------------------")
            print(f"Evaluating fairness by {disp}...")
        if args.plot_grouped_bar:
            group_metrics = {
                "FPR (1-Sensitivity)": false_positive_rate,
                "FNR (1-Specificity)": false_negative_rate,
                "Selection Rate": selection_rate,
                "Subject Count": count,
            }

        # Global fairness measures
        dpr, eor, eop = get_bootstrapped_fairness_measures(y_test, y_hat, attr_pf,
                                                           n_boot=args.boot_samples,
                                                           seed=args.seed,
                                                           verbose=args.verbose)
        print(f"Demographic Parity: {dpr[0]:.3f} (95% CI: [{dpr[1]:.3f}, {dpr[2]:.3f}])")
        print(f"Equalized Odds: {eor[0]:.3f} (95% CI: [{eor[1]:.3f}, {eor[2]:.3f}])")
        print(f"Equal Opportunity: {eop[0]:.3f} (95% CI: [{eop[1]:.3f}, {eop[2]:.3f}])")
        # Append measures to risk dictionary
        risk_dict['fair_' + disp] = {
            "DPR": round(dpr[0], 3),
            "DPR_CI": [round(dpr[1], 3), round(dpr[2], 3)],
            "EQO": round(eor[0], 3),
            "EQO_CI": [round(eor[1], 3), round(eor[2], 3)],
            "EOP": round(eop[0], 3),
            "EOP_CI": [round(eop[1], 3), round(eop[2], 3)],
        }
    
    save_pickle(risk_dict, fair_path, f"pf_{args.model_path}.pkl")
    print(f"Saved results dictionary to {fair_path}/pf_{args.model_path}.pkl")
    print('Evaluation complete.')