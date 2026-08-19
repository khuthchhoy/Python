"""Deep Temporal PyTorch Sequence Model with Additive Temporal Attention."""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from stock_predictor.config import DEFAULT_CONFIG, PredictionConfig
from stock_predictor.models.base import BaseStockModel

logger = logging.getLogger(__name__)

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    # Limit OpenMP threads to 1 on macOS arm64 to prevent thread contention segfaults
    torch.set_num_threads(1)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class TemporalAttention(nn.Module):
        """
        Learned Additive Temporal Attention mechanism over lookback horizon.
        Computes dynamic importance weights for each time-step in the sequence.
        """
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.attn = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # x shape: (batch_size, seq_len, hidden_dim)
            scores = self.attn(x)  # (batch_size, seq_len, 1)
            weights = torch.softmax(scores, dim=1)
            context = torch.sum(x * weights, dim=1)  # (batch_size, hidden_dim)
            return context, weights

    class PyTorchTemporalAttentionNet(nn.Module):
        """
        End-to-end Deep Temporal Sequence Model:
        1. Recurrent Temporal Network (GRU/LSTM) processing lookback sequences
        2. Temporal Attention module dynamically weighting historical days
        3. Multi-task heads: expected 5-day log-return and win probability P(Up)
        """
        def __init__(self, input_dim: int, hidden_dim: int = 48, num_layers: int = 1, dropout: float = 0.1):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim

            # Recurrent GRU layer for sequence modeling
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True
            )
            self.dropout = nn.Dropout(dropout)

            # Temporal attention over lookback steps
            self.attention = TemporalAttention(hidden_dim)

            # Point return regression head
            self.return_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )

            # Directional movement classifier head
            self.direction_head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # x: (batch_size, seq_len, input_dim)
            rnn_out, _ = self.rnn(x)
            rnn_out = self.dropout(rnn_out)
            context, weights = self.attention(rnn_out)
            pred_return = self.return_head(context).squeeze(-1)
            pred_prob = self.direction_head(context).squeeze(-1)
            return pred_return, pred_prob, weights


class LSTMStockModel(BaseStockModel):
    """
    Deep Neural Time-Series Sequence Predictor with PyTorch Temporal Attention.
    Extracts 3D sliding lookback sequences (T=20 days lookback) and learns dynamic temporal features.
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or DEFAULT_CONFIG
        self.scaler = RobustScaler()
        self.seq_len = self.config.sequence_length
        self.device = torch.device("cpu") if HAS_TORCH else "cpu"
        self.model: Optional[Any] = None
        self.feature_names: List[str] = []
        self._residual_std = 0.04
        self._last_attention_weights: Optional[np.ndarray] = None
        self._fallback_reg = None
        self._fallback_clf = None

    def _create_3d_sequences(
        self,
        X_arr: np.ndarray,
        y_arr: Optional[np.ndarray] = None,
        y_dir_arr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Create 3D sliding sequences of shape (samples, seq_len, n_features)."""
        num_samples, n_features = X_arr.shape
        if num_samples < self.seq_len:
            pad_count = self.seq_len - num_samples
            pad_x = np.repeat(X_arr[[0]], pad_count, axis=0)
            X_arr = np.vstack([pad_x, X_arr])
            if y_arr is not None:
                pad_y = np.repeat(y_arr[0], pad_count)
                y_arr = np.concatenate([pad_y, y_arr])
            if y_dir_arr is not None:
                pad_dir = np.repeat(y_dir_arr[0], pad_count)
                y_dir_arr = np.concatenate([pad_dir, y_dir_arr])
            num_samples = len(X_arr)

        seqs = []
        target_rets = []
        target_dirs = []

        for i in range(self.seq_len, num_samples + 1):
            window = X_arr[i - self.seq_len:i]
            seqs.append(window)
            if y_arr is not None:
                target_rets.append(y_arr[i - 1])
            if y_dir_arr is not None:
                target_dirs.append(y_dir_arr[i - 1])

        seqs_np = np.array(seqs, dtype=np.float32)
        rets_np = np.array(target_rets, dtype=np.float32) if y_arr is not None else None
        dirs_np = np.array(target_dirs, dtype=np.float32) if y_dir_arr is not None else None

        return seqs_np, rets_np, dirs_np

    def fit(
        self,
        X: pd.DataFrame,
        y_return: pd.Series,
        y_dir: Optional[pd.Series] = None,
        val_X: Optional[pd.DataFrame] = None,
        val_y: Optional[pd.Series] = None
    ) -> "LSTMStockModel":
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        y_arr = y_return.values.astype(np.float32)
        if y_dir is None:
            y_dir = (y_return > 0).astype(float)
        y_dir_arr = y_dir.values.astype(np.float32)

        seqs, rets, dirs = self._create_3d_sequences(X_scaled, y_arr, y_dir_arr)
        if len(seqs) == 0:
            return self

        if HAS_TORCH:
            try:
                torch.manual_seed(self.config.random_state)
                input_dim = seqs.shape[2]
                hidden_dim = self.config.lstm_hidden_dim
                
                self.model = PyTorchTemporalAttentionNet(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=self.config.lstm_dropout
                ).to(self.device)

                dataset = TensorDataset(
                    torch.from_numpy(seqs),
                    torch.from_numpy(rets),
                    torch.from_numpy(dirs)
                )

                batch_size = max(8, min(self.config.lstm_batch_size, len(seqs)))
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

                optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lstm_learning_rate, weight_decay=1e-4)
                criterion_ret = nn.HuberLoss(delta=0.02)
                criterion_dir = nn.BCELoss()

                epochs = max(5, min(self.config.lstm_epochs, 20))
                self.model.train()

                for epoch in range(epochs):
                    for b_x, b_y_ret, b_y_dir in loader:
                        b_x = b_x.to(self.device)
                        b_y_ret = b_y_ret.to(self.device)
                        b_y_dir = b_y_dir.to(self.device)

                        optimizer.zero_grad()
                        pred_ret, pred_dir, _ = self.model(b_x)
                        loss_ret = criterion_ret(pred_ret, b_y_ret)
                        loss_dir = criterion_dir(pred_dir, b_y_dir)
                        loss = loss_ret + 0.3 * loss_dir

                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        optimizer.step()

                # Calculate empirical residual standard deviation
                self.model.eval()
                with torch.no_grad():
                    all_x = torch.from_numpy(seqs).to(self.device)
                    all_preds, _, weights = self.model(all_x)
                    all_preds_np = all_preds.cpu().numpy()
                    residuals = rets - all_preds_np
                    self._residual_std = float(np.std(residuals)) if len(residuals) > 0 else 0.04
                    self._last_attention_weights = weights[-1].cpu().numpy().flatten()

                return self
            except Exception as e:
                logger.warning(f"PyTorch training failed ({e}). Falling back to Scikit-Learn MLP.")

        # Fallback to Scikit-Learn MLP
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        flat_seqs = seqs.reshape(len(seqs), -1)
        self._fallback_reg = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            max_iter=40,
            random_state=self.config.random_state
        )
        self._fallback_reg.fit(flat_seqs, rets)

        self._fallback_clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=40,
            random_state=self.config.random_state
        )
        self._fallback_clf.fit(flat_seqs, dirs.astype(int))

        preds = self._fallback_reg.predict(flat_seqs)
        self._residual_std = float(np.std(rets - preds)) if len(rets) > 0 else 0.04
        return self

    def predict_returns(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        seqs, _, _ = self._create_3d_sequences(X_scaled)
        if len(seqs) == 0:
            return np.zeros(len(X))

        if self.model is not None and HAS_TORCH:
            self.model.eval()
            with torch.no_grad():
                tensor_x = torch.from_numpy(seqs).to(self.device)
                pred_ret, _, _ = self.model(tensor_x)
                preds = pred_ret.cpu().numpy()
        elif self._fallback_reg is not None:
            flat_seqs = seqs.reshape(len(seqs), -1)
            preds = self._fallback_reg.predict(flat_seqs)
        else:
            return np.zeros(len(X))

        # Align length if padded or shorter than X
        if len(preds) < len(X):
            diff = len(X) - len(preds)
            preds = np.pad(preds, (diff, 0), mode="edge")
        elif len(preds) > len(X):
            preds = preds[-len(X):]

        return preds

    def predict_intervals(
        self,
        X: pd.DataFrame,
        quantiles: Tuple[float, float] = (0.10, 0.90)
    ) -> Tuple[np.ndarray, np.ndarray]:
        preds = self.predict_returns(X)
        z_low = -1.28  # 10th percentile
        z_high = 1.28  # 90th percentile
        lower = preds + z_low * self._residual_std
        upper = preds + z_high * self._residual_std
        upper = np.maximum(upper, lower + 1e-4)
        return lower, upper

    def predict_direction_prob(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        seqs, _, _ = self._create_3d_sequences(X_scaled)
        if len(seqs) == 0:
            return np.full(len(X), 0.5)

        if self.model is not None and HAS_TORCH:
            self.model.eval()
            with torch.no_grad():
                tensor_x = torch.from_numpy(seqs).to(self.device)
                _, pred_prob, _ = self.model(tensor_x)
                probs = pred_prob.cpu().numpy()
        elif self._fallback_clf is not None:
            flat_seqs = seqs.reshape(len(seqs), -1)
            try:
                probs_raw = self._fallback_clf.predict_proba(flat_seqs)
                probs = probs_raw[:, 1] if probs_raw.shape[1] == 2 else np.full(len(flat_seqs), 0.5)
            except Exception:
                probs = np.full(len(flat_seqs), 0.5)
        else:
            return np.full(len(X), 0.5)

        if len(probs) < len(X):
            diff = len(X) - len(probs)
            probs = np.pad(probs, (diff, 0), mode="edge")
        elif len(probs) > len(X):
            probs = probs[-len(X):]

        return np.clip(probs, 0.05, 0.95)

    def get_temporal_attention_weights(self) -> Optional[np.ndarray]:
        """Returns normalized temporal attention weights for the lookback sequence."""
        return self._last_attention_weights
