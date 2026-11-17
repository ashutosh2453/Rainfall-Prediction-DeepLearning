import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from models.cnn import CNNBaseline
from models.cnn_bilstm import CNNBiLSTM
from models.cnn_bilstm_attention import CNNBiLSTMAttention
from utils.dataset import RainfallDataset
from utils.early_stopping import EarlyStopping
from utils.metrics import evaluate_model, persistence_forecast, rain_skill_scores, skill_vs_persistence
from utils.trainer import fit, predict, train_one_epoch, validate

torch.manual_seed(0)
H, W, T = 12, 16, 7


def synthetic(n=24, seed=0):
    """Rain blobs that drift one cell east per day, so the next frame is predictable."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, T, H, W), dtype=np.float32)
    y = np.zeros((n, H, W), dtype=np.float32)
    for i in range(n):
        r, c = rng.integers(2, H - 3), rng.integers(0, W - T - 3)
        for t in range(T + 1):
            frame = np.zeros((H, W), dtype=np.float32)
            frame[r:r + 3, c + t:c + t + 3] = 1.0
            if t < T:
                X[i, t] = frame
            else:
                y[i] = frame
    return X, y


# ---------------------------------------------------------------- dataset & models

def test_dataset_adds_channel_dims():
    X, y = synthetic(4)
    xb, yb = RainfallDataset(X, y)[0]
    assert xb.shape == (T, 1, H, W) and yb.shape == (1, H, W)


@pytest.mark.parametrize("model", [CNNBaseline(), CNNBiLSTM(H, W), CNNBiLSTMAttention(H, W)])
def test_models_map_sequences_to_next_frame(model):
    x = torch.randn(3, T, 1, H, W)
    assert model(x).shape == (3, 1, H, W)


def test_default_grid_is_unchanged_for_the_notebooks():
    assert CNNBiLSTM().height == 129 and CNNBiLSTMAttention().width == 135


def test_attention_weights_sum_to_one():
    model = CNNBiLSTMAttention(H, W)
    seq = torch.randn(2, T, 256)
    weights = torch.softmax(model.attention.attention(seq), dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2, 1))


# ---------------------------------------------------------------- early stopping

def test_early_stopping_is_backward_compatible():
    es = EarlyStopping(patience=2)
    for loss in [1.0, 0.8, 0.9, 0.95]:
        es(loss)
    assert es.early_stop and es.best_loss == 0.8 and es.best_epoch == 2


def test_early_stopping_restores_best_weights():
    model = torch.nn.Linear(2, 1)
    es = EarlyStopping(patience=1)
    es(0.5, model)
    best = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(10.0)
    es(0.9, model)  # worse epoch: weights changed but must not be kept
    es.restore_best(model)
    assert all(torch.equal(best[k], v) for k, v in model.state_dict().items())


def test_early_stopping_stops_on_divergence():
    es = EarlyStopping(patience=10)
    es(float("nan"))
    assert es.early_stop
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


# ---------------------------------------------------------------- training

def test_losses_are_sample_weighted():
    X, y = synthetic(5)
    loader = DataLoader(RainfallDataset(X, y), batch_size=4)  # batches of 4 and 1
    model, crit = CNNBaseline(), torch.nn.MSELoss()
    model.eval()
    with torch.no_grad():
        full = crit(model(torch.tensor(X).unsqueeze(2)), torch.tensor(y).unsqueeze(1)).item()
    assert validate(model, loader, crit, "cpu") == pytest.approx(full, rel=1e-5)


def test_fit_learns_drift_and_beats_persistence():
    X, y = synthetic(48)
    tr = DataLoader(RainfallDataset(X[:36], y[:36]), batch_size=8, shuffle=True)
    va = DataLoader(RainfallDataset(X[36:], y[36:]), batch_size=8)
    model = CNNBaseline()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    es = EarlyStopping(patience=5)
    hist = fit(model, tr, va, opt, torch.nn.MSELoss(), "cpu", epochs=60, early_stopping=es, verbose=False)
    assert hist["val_loss"][hist["best_epoch"] - 1] == min(hist["val_loss"])
    assert validate(model, va, torch.nn.MSELoss(), "cpu") == pytest.approx(es.best_loss, rel=1e-5)
    y_true, y_pred = predict(model, va, "cpu")
    skill = skill_vs_persistence(y_true, y_pred, X[36:])
    assert skill["skill_score"] > 0, skill  # the model must beat "tomorrow = today"


def test_gradient_clipping_option_runs():
    X, y = synthetic(8)
    loader = DataLoader(RainfallDataset(X, y), batch_size=4)
    model = CNNBiLSTM(H, W)
    loss = train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.1), torch.nn.MSELoss(), "cpu", max_grad_norm=0.5)
    assert math.isfinite(loss)


# ---------------------------------------------------------------- metrics

def test_evaluate_model_with_mask():
    y = np.array([[1.0, 2.0], [0.0, 0.0]])
    p = np.array([[1.0, 4.0], [5.0, 5.0]])
    mask = np.array([[True, True], [False, False]])
    m = evaluate_model(y, p, mask)
    assert m["MAE"] == pytest.approx(1.0) and m["MSE"] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_model(y, p[:1])


def test_rain_skill_scores():
    obs = np.array([5, 5, 0, 0, 5, 0], dtype=float)
    fc = np.array([5, 0, 5, 0, 5, 0], dtype=float)
    s = rain_skill_scores(obs, fc, threshold=1.0)
    assert (s["hits"], s["misses"], s["false_alarms"]) == (2, 1, 1)
    assert s["POD"] == pytest.approx(2 / 3) and s["FAR"] == pytest.approx(1 / 3)
    assert s["CSI"] == pytest.approx(0.5) and s["Bias"] == pytest.approx(1.0)
    assert math.isnan(rain_skill_scores(np.zeros(3), np.zeros(3), 1.0)["POD"])


def test_persistence_forecast_shapes():
    X = np.arange(2 * T * 2 * 2, dtype=float).reshape(2, T, 2, 2)
    p = persistence_forecast(X)
    assert p.shape == (2, 1, 2, 2) and np.array_equal(p[:, 0], X[:, -1])
    assert persistence_forecast(X[:, :, None]).shape == (2, 1, 2, 2)
    perfect = skill_vs_persistence(X[:, -1:], X[:, -1:], X)
    assert math.isnan(perfect["skill_score"])  # persistence is already perfect here
