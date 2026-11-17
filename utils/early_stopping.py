import copy

import numpy as np


class EarlyStopping:
    """Stop training when validation loss stops improving, and remember the best weights.

    Backwards compatible with the notebooks: ``early_stopping(val_loss)`` still works and
    ``early_stop`` / ``best_loss`` behave as before. Pass the model as well
    (``early_stopping(val_loss, model)``) to keep a copy of the best weights, then call
    ``restore_best(model)`` after training. Without this, training ends with the weights
    from the last (worse) epochs.
    """

    def __init__(self, patience=5, delta=0):
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.delta = delta

        self.counter = 0
        self.best_loss = np.inf
        self.best_epoch = None
        self.best_state = None
        self.early_stop = False
        self._epoch = 0

    def __call__(self, val_loss, model=None):
        if not np.isfinite(val_loss):
            # A NaN/inf loss means training diverged: stop rather than keep going.
            self.early_stop = True
            return False

        self._epoch += 1
        improved = val_loss < self.best_loss - self.delta
        if improved:
            self.best_loss = val_loss
            self.best_epoch = self._epoch
            self.counter = 0
            if model is not None:
                self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return improved

    def restore_best(self, model):
        """Load the best weights seen so far into ``model`` (no-op if none were recorded)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
