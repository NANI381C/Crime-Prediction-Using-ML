# src/shap_explainer.py
import shap
import numpy as np
import matplotlib.pyplot as plt
import joblib
import tempfile
import os

# KernelExplainer can be slow — we use a small background sample
def explain_lstm_with_shap(model, X_train_scaled, feature_names, scaler_y=None, nsamples=100):
    """
    model: a trained keras model that accepts input shape (n, window, features)
    X_train_scaled: numpy array shaped (n_samples, window, features) - scaled
    feature_names: list of feature names (string) corresponding to features per timestep
    scaler_y: optional scaler for inverse transform (not used in SHAP)
    nsamples: background sample size for KernelExplainer
    Returns: path to saved shap summary image (png)
    """
    # Flatten sequences by averaging across window — KernelExplainer expects 2D features.
    # This is a compromise: treat each feature's average over the window as input.
    X_flat = X_train_scaled.mean(axis=1)  # shape (n_samples, features)

    # Use a small background dataset
    background = X_flat[np.random.choice(X_flat.shape[0], min(nsamples, X_flat.shape[0]), replace=False)]

    # Define a wrapper prediction function that Accepts 2D input (n_samples, features)
    def predict_fn(X2d):
        # X2d -> expand back to sequences by repeating across window
        window = X_train_scaled.shape[1]
        Xseq = np.repeat(X2d[:, np.newaxis, :], window, axis=1)
        preds = model.predict(Xseq, verbose=0)
        return preds.flatten()

    explainer = shap.KernelExplainer(predict_fn, background, link="identity")

    # pick a small set to explain
    to_explain = X_flat[np.random.choice(X_flat.shape[0], min(50, X_flat.shape[0]), replace=False)]

    shap_values = explainer.shap_values(to_explain, nsamples=200)

    # Plot summary (bar)
    plt.figure(figsize=(8,6))
    shap.summary_plot(shap_values, to_explain, feature_names=feature_names, show=False, plot_type="bar")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name, bbox_inches="tight", dpi=150)
    plt.close()
    return tmp.name
