import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(y_true, y_pred):

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        y_pred
    )

    return {

        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2

    }


def print_metrics(metrics):

    print("=" * 40)

    for key, value in metrics.items():

        print(f"{key:<8}: {value:.6f}")

    print("=" * 40)