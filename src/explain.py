import shap
# TODO: Implement SHAP evaluator module for measuring multimodal feature importance
raise NotImplementedError
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal performance evaluation pipeline.")
    parser.add_argument(
            "--test", default=None, help="List of ids to use for evaluation."
    )
    parser.add_argument(
            "--explain",
            action="store_true",
            help="Whether to generate explainability plots.",
    )

    if model_type == "rf":
            # Visualise important features
            features = test_set.get_feature_list()
            importances = model.feature_importances_
            indices = np.argsort(importances)

            plt.figure(figsize=(20, 10))
            plt.title("Feature Importances")
            plt.barh(
                range(len(indices)), importances[indices], color="b", align="center"
            )
            plt.yticks(range(len(indices)), [features[i] for i in indices])
            plt.xlabel("Relative Importance")
            plt.show()

            # Create shap plot
            explainer = shap.TreeExplainer(model, x_test)

            # Plot single waterfall plot
            # Correct classification (TP)
            tps = np.argwhere(np.logical_and(y_hat == 1, y_test == 1))

            if len(tps) > 0:
                tp = tps[0][0]

                plt.figure(figsize=(12, 4))
                plt.title(
                    f"Truth: {int(y_test[tp])}, Predict: {int(y_hat[tp])}, Prob: {round(prob[tp], 2)}"
                )
                shap.bar_plot(
                    explainer(x_test[tp])[:, 1].values,
                    feature_names=features,
                    max_display=20,
                )
                plt.show()

            # Incorrect (FN)
            fns = np.argwhere(np.logical_and(y_hat == 0, y_test == 1))

            if len(fns) > 0:
                fn = fns[0][0]

                plt.figure(figsize=(12, 4))
                plt.title(
                    f"Truth: {int(y_test[fn])}, Predict: {int(y_hat[fn])}, Prob: {round(prob[fn], 2)}"
                )
                shap.bar_plot(
                    explainer(x_test[fn])[:, 1].values,
                    feature_names=features,
                    max_display=20,
                )
                plt.show()

            # Plot summary over all test subjects
            start = time.time()
            shap_values = explainer(x_test, check_additivity=False)
            print(time.time() - start)

            plt.figure()
            shap.summary_plot(
                shap_values[:, :, 1], feature_names=features, max_display=20
            )
            plt.show()

    elif model_type == "fusion":
        # get first collated batch (fixed size and num of samples)
        batch = next(iter(test_dataloader))

        for i in range(2):
                features = test_set.get_feature_list(f"dynamic_{i}")

                x_test = batch[2][i]
                explainer = shap.DeepExplainer(model.embed_timeseries[i], x_test)

                # Plot summary over all test subjects for single timepoint (t=0)
                shap_values = explainer.shap_values(x_test, check_additivity=False)

                plt.figure()
                shap.summary_plot(
                shap_values.mean(axis=3)[:, 0, :],
                feature_names=features,
                features=x_test[:, 0, :],
                )
                plt.show()