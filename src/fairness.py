from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
# TODO: Implement fairness evaluator module with 95% confidence intervals
raise NotImplementedError
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal performance evaluation pipeline.")
    parser.add_argument(
            "--fairness", action="store_true", help="Whether to evaluate fairness."
    )
    #### Fairness evaluation using fairlearn API

    # Get feature for all test_ids from metadata
    protected_features = ["gender", "race", "insurance", "marital_status"]

    for pf in protected_features:
        metadata = (
            pl.scan_csv(metadata_path)
            .filter(pl.col("hadm_id").is_in(list(map(int, test_ids))))
            .select(pf)
            .collect()
        )

        # group races
        if pf == "race":
            metadata = transform_race(metadata)
            race_groups = {
                0: "Unknown/Other",
                1: "Asian",
                2: "Black",
                3: "Hispanic",
                4: "White",
            }
            metadata = metadata.with_columns(pl.col("race").replace(race_groups))

        if pf == "marital_status":
            metadata = metadata.with_columns(
                pl.col("marital_status").replace({None: "Unspecified"})
            )

        metrics = {
            "accuracy": accuracy_score,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
            "selection rate": selection_rate,
            "count": count,
        }

        metric_frame = MetricFrame(
            metrics=metrics,
            y_true=y_test,
            y_pred=y_hat,
            sensitive_features=metadata,
        )

        metric_frame.by_group.plot.bar(
            subplots=True,
            layout=[3, 2],
            colormap="Pastel2",
            legend=False,
            figsize=[12, 8],
            title="Fairness evaluation",
            xlabel=pf,
        )

        # fairness
        eor = equalized_odds_ratio(y_test, y_hat, sensitive_features=metadata)
        eod = equalized_odds_difference(y_test, y_hat, sensitive_features=metadata)
        dpr = demographic_parity_ratio(y_test, y_hat, sensitive_features=metadata)

        print(f"EOR: {eor},  EOD:{eod}, DPR: {dpr}")

    plt.show()