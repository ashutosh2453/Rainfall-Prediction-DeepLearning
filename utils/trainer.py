import numpy as np
import torch


def _batch_size(x):
    return x.size(0)


def train_one_epoch(model, loader, optimizer, criterion, device, max_grad_norm=None):
    """One training epoch. Returns the loss averaged over *samples*, so a small final
    batch no longer counts as much as a full one. ``max_grad_norm`` enables gradient
    clipping, which stabilises the LSTM models."""
    model.train()
    total, n = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item() * _batch_size(X)
        n += _batch_size(X)
    return total / max(n, 1)


def validate(model, loader, criterion, device):
    """Validation loss averaged over samples."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            loss = criterion(model(X), y)
            total += loss.item() * _batch_size(X)
            n += _batch_size(X)
    return total / max(n, 1)


def predict(model, loader, device):
    """Return (y_true, y_pred) as numpy arrays for a whole loader."""
    model.eval()
    trues, preds = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.append(model(X.to(device)).cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(trues), np.concatenate(preds)


def fit(model, train_loader, val_loader, optimizer, criterion, device, epochs=50, early_stopping=None,
        scheduler=None, max_grad_norm=1.0, verbose=True):
    """Full training loop: early stopping on validation loss, best-weight restoration,
    optional LR scheduler (ReduceLROnPlateau is stepped with the validation loss).

    Returns a history dict with per-epoch train/val losses and the best epoch.
    """
    history = {"train_loss": [], "val_loss": [], "best_epoch": None}
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm)
        va = validate(model, val_loader, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(va)
            else:
                scheduler.step()
        if verbose:
            print(f"epoch {epoch:3d}  train {tr:.6f}  val {va:.6f}")
        if early_stopping is not None:
            early_stopping(va, model)
            if early_stopping.early_stop:
                if verbose:
                    print(f"early stopping at epoch {epoch}; best epoch {early_stopping.best_epoch}")
                break
    if early_stopping is not None:
        early_stopping.restore_best(model)
        history["best_epoch"] = early_stopping.best_epoch
    return history
