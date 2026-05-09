"""
ml_engine.py — sklearn ML models: Linear Regression, SVM, KNN
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


FEATURE_COLS = [
    "subsystem_id", "connection_count",
    "est_size_cm",  "is_actuated", "label_area_pct",
]

FEATURE_LABELS = {
    "subsystem_id":    "Subsystem category",
    "connection_count":"No. of connections",
    "est_size_cm":     "Estimated size (cm)",
    "is_actuated":     "Requires actuation",
    "label_area_pct":  "Diagram area %",
}


class MLEngine:

    def run(self, df: pd.DataFrame, model_name: str, target: str, test_size: float) -> dict:
        # ── Prepare features ────────────────────────────────────────────────
        available = [c for c in FEATURE_COLS if c in df.columns and c != target]
        if target not in df.columns:
            return {"error": f"Target column '{target}' not found in data."}
        if len(available) < 2:
            return {"error": "Not enough feature columns."}

        X = df[available].fillna(0).astype(float)
        y = df[target].astype(float)

        if len(df) < 6:
            return {"error": "Need at least 6 components for ML. Upload a richer diagram."}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # ── Choose model ────────────────────────────────────────────────────
        if "Linear" in model_name:
            model = LinearRegression()
            model.fit(X_train_s, y_train)
        elif "SVM" in model_name:
            model = SVR(kernel="rbf", C=10, epsilon=0.2)
            model.fit(X_train_s, y_train)
        else:  # KNN
            k = max(2, min(5, len(X_train) - 1))
            model = KNeighborsRegressor(n_neighbors=k)
            model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)

        # ── Metrics ────────────────────────────────────────────────────────
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        # Cross-val on full data
        cv_scores = cross_val_score(
            model, scaler.fit_transform(X), y,
            cv=min(5, len(df) // 2), scoring="r2"
        )

        metrics = {"R2": round(r2, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4),
                   "CV_R2_mean": round(cv_scores.mean(), 4), "CV_R2_std": round(cv_scores.std(), 4)}

        # ── Predictions table ───────────────────────────────────────────────
        names = df.loc[y_test.index, "name"].values if "name" in df.columns else y_test.index.astype(str).values
        errors = y_test.values - y_pred
        err_pct = np.where(y_test.values != 0, np.abs(errors / y_test.values) * 100, 0)

        pred_df = pd.DataFrame({
            "Component":  names,
            "Actual":     y_test.values.round(2),
            "Predicted":  y_pred.round(2),
            "Error":      errors.round(2),
            "Error%":     err_pct.round(1),
        })

        # ── Feature importance ──────────────────────────────────────────────
        fi = None
        if hasattr(model, "coef_"):
            fi = [
                {"Feature": FEATURE_LABELS.get(f, f), "Importance": round(abs(c), 4), "Direction": "↑" if c > 0 else "↓"}
                for f, c in zip(available, model.coef_)
            ]
        else:
            # Permutation-style importance via correlation
            fi = [
                {"Feature": FEATURE_LABELS.get(f, f),
                 "Importance": round(abs(df[f].corr(df[target])), 4),
                 "Direction": "↑" if df[f].corr(df[target]) > 0 else "↓"}
                for f in available
            ]
        fi.sort(key=lambda x: x["Importance"], reverse=True)

        # ── Coefficients (LR only) ──────────────────────────────────────────
        coefs = None
        if hasattr(model, "coef_") and "Linear" in model_name:
            coefs = [
                {"Feature": FEATURE_LABELS.get(f, f),
                 "Coefficient": round(c, 4),
                 "Interpretation": _interpret_coef(f, c)}
                for f, c in zip(available, model.coef_)
            ]
            coefs.append({"Feature": "Intercept", "Coefficient": round(model.intercept_, 4), "Interpretation": "Baseline prediction"})

        return {
            "model_name": model_name,
            "target": target,
            "metrics": metrics,
            "predictions_df": pred_df,
            "feature_importance": fi,
            "coefficients": coefs,
            "samples": {"train": len(X_train), "test": len(X_test)},
            "cv_scores": cv_scores.round(4).tolist(),
        }


def _interpret_coef(feature: str, coef: float) -> str:
    direction = "increases" if coef > 0 else "decreases"
    mag = abs(coef)
    strength = "strongly" if mag > 1 else ("moderately" if mag > 0.4 else "slightly")
    labels = {
        "subsystem_id":    f"Being in a higher-numbered subsystem {strength} {direction} the target",
        "connection_count":f"Each extra connection {strength} {direction} the target by {mag:.2f}",
        "est_size_cm":     f"Each cm increase in size {strength} {direction} the target by {mag:.2f}",
        "is_actuated":     f"Actuated components {strength} {direction} the target by {mag:.2f}",
        "label_area_pct":  f"Larger diagram footprint {strength} {direction} the target by {mag:.2f}",
    }
    return labels.get(feature, f"Coefficient: {coef:.4f}")
