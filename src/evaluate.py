import argparse
import os
import glob
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import toml
from datasets import CollateFn, CollateTimeSeries, MIMIC4Dataset

from lightning.pytorch import Trainer

from models import MMModel
from torch import concat
from torch.utils.data import DataLoader
from utils.eval_utils import get_all_roc_pr_summary, get_pr_performance, get_roc_performance, plot_calibration_curve, plot_learning_curve, plot_pr, plot_roc, rank_prediction_deciles
from utils.functions import load_pickle, save_pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal performance evaluation pipeline.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the pickled data dictionary generated from prepare_data.py.",
        default="../outputs/prep_data/mmfair_feat.pkl",
    )
    parser.add_argument(
        "col_path",
        type=str,
        help="Path to the pickled column dictionary generated from prepare_data.py.",
        default="../outputs/prep_data/mmfair_cols.pkl",
    )
    parser.add_argument(
        "ids_path",
        type=str,
        help="Directory containing train/val/test ids.",
        default="../outputs/prep_data",
    )
    parser.add_argument(
        "attr_path",
        type=str,
        help="Directory containing attributes metadata (original ehr_static.csv).",
        default="../outputs/ext_data/ehr_static.csv",
    )
    parser.add_argument(
        "eval_path",
        type=str,
        help="Directory to store performance plots.",
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
        "model_path",
        type=str,
        help="Path to the saved model.",
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
        "--n_bins",
        type=int,
        default=10,
        help="Number of bins for decile analysis and calibration curve.",
    )
    parser.add_argument(
        "--strat_by_attr",
        action="store_true",
        help="Show stratified decile analysis by attribute.",
    )
    parser.add_argument(
        "--all_outcomes",
        action="store_true",
        help="Generate Precision-Recall summary across outcomes in targets.toml."
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        default=False,
        help="Control verbosity. If true, will make more .collect() calls to compute dataset size.",
    )

    args = parser.parse_args()
    config = toml.load(args.config)
    targets = toml.load(args.targets)
    fusion_method = config["model"]["fusion_method"]
    modalities = config["data"]["modalities"]
    static_only = True if (len(modalities) == 1) and ("static" in modalities) else False
    with_notes = True if "notes" in modalities else False
    batch_size = config["data"]["batch_size"]
    mod_str = "_".join(modalities)
    ### General setup
    outcomes = targets["outcomes"]["labels"]
    outcomes_disp = targets["outcomes"]["display"]
    outcomes_col = targets["outcomes"]["colormap"]
    attributes = targets["attributes"]["labels"]
    attr_disp = targets["attributes"]["display"]
    ### For Pre-loading models if using --all_outcomes
    paths = config["paths"]["model_paths"]
    if args.outcome not in outcomes:
        print(f"Outcome {args.outcome} must be included in targets.toml.")
        sys.exit()
    outcome_idx = outcomes.index(args.outcome)
    ### Set plotting style
    plt.rcParams.update(
        {"font.size": 12, "font.weight": "normal", "font.family": "serif"}
    )
    print('------------------------------------------')
    print("MMHealthFair: Multimodal evaluation pipeline")
    ### If --all_outcomes, load results dictionary and generate summary
    if args.all_outcomes:
        print("Generating Precision-Recall summary across all outcomes...")
        res_all = {}
        for outcome, path in zip(outcomes, args.paths):
            res_dict = load_pickle(path)
            res_all[outcome] = res_dict
        get_all_roc_pr_summary(res_all, outcomes_disp, outcomes_col,
                               output_roc_path=f"{args.eval_path}/roc_full_{args.fusion_method}_{mod_str}.png",
                               output_pr_path=f"{args.eval_path}/pr_full_{args.fusion_method}_{mod_str}.png")
        print("Evaluation complete.")
        sys.exit()
    print(f'Evaluating performance for outcome "{outcomes_disp[outcome_idx]}"')
    print(f'Modalities used: {modalities}')
    print(f'Fusion method: {fusion_method}')
    print('------------------------------------------')

    # if loading from .ckpt (deep learning) set static_only to False else assume static only model (RF)
    #model_type = "fusion" if os.path.splitext(args.model_path)[1] == ".ckpt" else "rf"
    #static_only = True if model_type == "rf" else False

    # Get test ids
    if len(glob.glob(os.path.join(args.ids_path, "test_ids_" + {args.outcome} + ".csv"))) == 0:
        print(f"No test ids found for outcome {args.outcome}. Exiting..")
        sys.exit()
    test_ids = pl.read_csv(os.path_join(args.ids_path, "test_ids_" + {args.outcome} + ".csv")).select("subject_id").to_numpy()

    if args.by_attribute:
        print("Reading attributes metadata for stratified decile analysis...")
        if not os.path.exists(args.attr_path):
            print("Attributes metadata not found. Exiting...")
            sys.exit()
        ehr_static = pl.read_csv(args.attr_path).to_pandas()

    test_set = MIMIC4Dataset(
        args.data_path,
        "test",
        ids=test_ids,
        static_only=static_only,
        with_notes=with_notes
    )
    test_set.print_label_dist()
    test_dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        collate_fn=CollateFn() if static_only else CollateTimeSeries()
    )

    model = MMModel.load_from_checkpoint(checkpoint_path=args.model_path)
    print("Evaluating on test data...")
    trainer = Trainer(accelerator="gpu")
    output = trainer.predict(model, dataloaders=test_dataloader)
    default_thresh = 0.5
    prob = concat([out[0] for out in output])
    y_hat = np.where(prob > default_thresh, 1, 0)
    y_test = concat([out[1] for out in output])

    ### Performance summary
    print('Plotting learning curve...')
    if not os.path.exists(args.eval_path):
        os.makedirs(args.eval_path)
    log_dir=f"logs/{args.outcome}_{args.fusion_method}_{mod_str}/losses.csv"
    plot_learning_curve(log_dir, output_path=f"{args.eval_path}/lc_{args.outcome}_{args.fusion_method}_{mod_str}.png")
    print('Getting ROC/PR curves...')
    plot_roc(y_test, prob, outcome=args.outcome, output_path=f"{args.eval_path}/roc_{args.outcome}_{args.fusion_method}_{mod_str}.png")
    plot_pr(y_test, prob, outcome=args.outcome, output_path=f"{args.eval_path}/pr_{args.outcome}_{args.fusion_method}_{mod_str}.png")
    print('ROC performance report:')
    print('------------------------------------------')
    bin_labels, res_roc_dict = get_roc_performance(y_test, prob, verbose=args.verbose)
    print('------------------------------------------')
    print('Precision-Recall performance report:')
    print('------------------------------------------')
    bin_labels, res_pr_dict = get_pr_performance(y_test, prob, bin_labels, 
                                              opt_f1=True, verbose=args.verbose)
    print('------------------------------------------')
    print('Plotting calibration curve...')
    plot_calibration_curve(y_test, prob, outcome=args.outcome, n_bins=args.n_bins,
                           output_path=f"{args.eval_path}/calib_{args.outcome}_{args.fusion_method}_{mod_str}.png")
    print('------------------------------------------')
    print('Prediction decile analysis:')
    print('------------------------------------------')
    res_pd_dict = rank_prediction_deciles(y_test, prob, args.n_bins,
                                          args.outcome, output_path=f"{args.eval_path}/rstrat_{args.outcome}_{args.fusion_method}_{mod_str}.png",
                                          by_attribute=args.strat_by_attr,
                                          attrs=attributes, attr_disp=attr_disp,
                                          test_ids=test_ids,
                                          attr_features=ehr_static,
                                          verbose=args.verbose)
    print('------------------------------------------')
    print('Saving results dictionary...')
    res_dict = res_roc_dict | res_pr_dict | res_pd_dict
    save_pickle(res_dict, args.eval_path, "performance_{args.outcome}_{args.fusion_method}_{mod_str}.pkl")
    print('Evaluation complete.')

    '''
    if model_type == "rf":
        print("Loading dataset...")
        x_test = []
        y_test = []
        for data in test_set:
            x, y = data
            x_test.append(x[0])
            y_test.append(y[0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        model = load_pickle(args.model_path)
        print("Evaluating on test data...")
        y_hat = model.predict(x_test)
        prob = model.predict_proba(x_test)[:, 1]'
    '''
