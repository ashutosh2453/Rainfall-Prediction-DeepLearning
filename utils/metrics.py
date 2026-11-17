import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(y_true, y_pred, mask=None):
    """MAE / MSE / RMSE / R² over all grid cells.

    ``mask`` (boolean, broadcastable to one map) restricts the scores to valid land
    cells, so ocean / missing-data cells that are always 0 don't inflate the scores.
    """
    y_true, y_pred = _flatten(y_true, y_pred, mask)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def rain_skill_scores(y_true, y_pred, threshold, mask=None):
    """Categorical verification of rain / no-rain at ``threshold`` (in the data's units).

    POD  = hits / (hits + misses)                 probability of detection
    FAR  = false alarms / (hits + false alarms)   false-alarm ratio
    CSI  = hits / (hits + misses + false alarms)  critical success index
    Bias = (hits + false alarms) / (hits + misses)  frequency bias (1 = unbiased)
    """
    y_true, y_pred = _flatten(y_true, y_pred, mask)
    obs, fc = y_true >= threshold, y_pred >= threshold
    hits = int(np.sum(obs & fc))
    misses = int(np.sum(obs & ~fc))
    false_alarms = int(np.sum(~obs & fc))
    return {
        "POD": _safe(hits, hits + misses),
        "FAR": _safe(false_alarms, hits + false_alarms),
        "CSI": _safe(hits, hits + misses + false_alarms),
        "Bias": _safe(hits + false_alarms, hits + misses),
        "hits": hits, "misses": misses, "false_alarms": false_alarms,
    }


def persistence_forecast(X):
    """Baseline "tomorrow looks like today": the last input frame, shaped like the targets.

    Accepts inputs shaped (N, T, H, W) or (N, T, 1, H, W) and returns (N, 1, H, W).
    """
    X = np.asarray(X)
    last = X[:, -1]
    return last if last.ndim == 4 else last[:, None]


def skill_vs_persistence(y_true, y_pred, X, mask=None):
    """MSE skill score against persistence: 1 is perfect, 0 is no better than persistence,
    and negative means the model is worse than simply repeating today's map."""
    model_mse = evaluate_model(y_true, y_pred, mask)["MSE"]
    base_mse = evaluate_model(y_true, persistence_forecast(X).reshape(np.shape(y_true)), mask)["MSE"]
    return {"model_MSE": model_mse, "persistence_MSE": base_mse,
            "skill_score": float(1 - model_mse / base_mse) if base_mse > 0 else float("nan")}


def print_metrics(metrics):
    print("=" * 40)
    for key, value in metrics.items():
        print(f"{key:<16}: {value:.6f}" if isinstance(value, float) else f"{key:<16}: {value}")
    print("=" * 40)


def _flatten(y_true, y_pred, mask):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if mask is not None:
        m = np.broadcast_to(np.asarray(mask, dtype=bool), y_true.shape)
        return y_true[m], y_pred[m]
    return y_true.reshape(-1), y_pred.reshape(-1)


def _safe(num, den):
    return float(num / den) if den else float("nan")
