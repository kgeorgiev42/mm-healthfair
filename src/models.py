import csv
import os

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.callbacks import Callback
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


# nn.Modules
class LSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers=1, hidden_dim=128, dropout=0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers, batch_first=True
        )
        self.project = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, input_):
        _, (h_T, _) = self.lstm(input_)
        output = self.dropout(self.project(h_T[-1]))
        return self.relu(output)


class Gate(nn.Module):
    # Adapted from https://github.com/emnlp-mimic/mimic/blob/main/base.py#L136 inspired by https://arxiv.org/pdf/1908.05787
    def __init__(self, inp1_size, inp2_size, inp3_size: int = 0, dropout: int = 0):
        super().__init__()

        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc2 = nn.Linear(inp1_size + inp3_size, 1)
        self.fc3 = nn.Linear(inp2_size + inp3_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2, inp3=None):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        if inp3 is not None:
            w3 = torch.sigmoid(self.fc2(torch.cat([inp1, inp3], -1)))
            adjust = self.fc3(torch.cat([w2 * inp2, w3 * inp3], -1))
        else:
            # only need to adjust input 2
            adjust = self.fc3(w2 * inp2)

        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output)).squeeze()
        return output


# lightning.LightningModules
class MMModel(L.LightningModule):
    def __init__(
        self,
        st_input_dim=18,
        st_embed_dim=64,
        ts_input_dim=(9, 7),
        ts_embed_dim=64,
        nt_input_dim=768,
        nt_embed_dim=64,
        num_layers=1,
        dropout=0.2,
        num_ts=2,
        target_size=1,
        lr=0.1,
        fusion_method="concat",
        st_first=True,
        modalities=None,
        with_packed_sequences=False,
        dataset=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.modalities = set(modalities)
        self.with_static = "static" in self.modalities
        self.with_ts = "timeseries" in self.modalities
        self.with_notes = "notes" in self.modalities
        self.fusion_method = fusion_method
        self.st_first = st_first

        # Static embedding
        self.embed_static = (
            nn.Sequential(
                nn.Linear(st_input_dim, st_embed_dim // 2),
                nn.LayerNorm(st_embed_dim // 2),
                nn.Linear(st_embed_dim // 2, st_embed_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
            )
            if self.with_static
            else None
        )
        st_embed_dim = st_embed_dim if self.with_static else 0

        # Time-series embedding
        self.embed_timeseries = (
            nn.ModuleList(
                [
                    LSTM(
                        ts_input_dim[i],
                        ts_embed_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                    )
                    for i in range(num_ts)
                ]
            )
            if self.with_ts
            else None
        )
        ts_embed_dim = ts_embed_dim if self.with_ts else 0

        # Notes embedding
        self.embed_notes = (
            nn.Sequential(
                nn.Linear(nt_input_dim, nt_embed_dim),
                nn.LayerNorm(nt_embed_dim),
                nn.ReLU(),
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=nt_embed_dim,
                        nhead=8,
                        dim_feedforward=256,
                        dropout=dropout,
                        activation="relu",
                        batch_first=True,
                    ),
                    num_layers=2,
                ),
                nn.Dropout(dropout),
            )
            if self.with_notes
            else None
        )
        nt_embed_dim = nt_embed_dim if self.with_notes else 0

        # Fusion and final layer
        if fusion_method == "mag" and self.with_ts and self.with_static:
            self.fuse = Gate(
                st_embed_dim if st_first else ts_embed_dim,
                *([ts_embed_dim] * num_ts if st_first else [st_embed_dim]),
                dropout=dropout,
            )
            self.fc = nn.Linear(st_embed_dim if st_first else ts_embed_dim, target_size)
        elif fusion_method == "concat":
            total_dim = st_embed_dim + (num_ts * ts_embed_dim) + nt_embed_dim
            self.fc = nn.Linear(total_dim, target_size)
        else:
            embed_dim = st_embed_dim or (num_ts * ts_embed_dim) or nt_embed_dim
            self.fc = nn.Linear(embed_dim, target_size)

        # Loss function with class weights
        pos_weight = self.compute_class_weights(dataset) if dataset else None
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Metrics
        self.lr = lr
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auc = torchmetrics.AUROC(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.ap = torchmetrics.AveragePrecision(task="binary")
        self.with_packed_sequences = with_packed_sequences

    def prepare_batch(self, batch):
        s, y, d, lengths, n = batch[0], batch[1], batch[2], batch[3], batch[4]
        st_embed = self.embed_static(s) if self.with_static else None
        ts_embed = (
            [
                self.embed_timeseries[i](
                    torch.nn.utils.rnn.pack_padded_sequence(
                        d[i], lengths[i], batch_first=True, enforce_sorted=False
                    )
                    if self.with_packed_sequences
                    else d[i]
                ).unsqueeze(1)
                for i in range(self.hparams.num_ts)
            ]
            if self.with_ts
            else None
        )
        nt_embed = self.embed_notes(n) if self.with_notes else None

        # Fuse embeddings
        if self.fusion_method == "concat":
            embeddings = [
                e for e in [st_embed, *(ts_embed or []), nt_embed] if e is not None
            ]
            out = torch.concat(embeddings, dim=-1).squeeze()
        elif self.fusion_method == "mag" and self.with_ts:
            out = (
                self.fuse(st_embed, *ts_embed)
                if self.st_first
                else self.fuse(*ts_embed, st_embed)
            )
        else:
            out = (
                st_embed
                or torch.cat([t.squeeze() for t in ts_embed], dim=-1)
                or nt_embed
            ).squeeze()

        x_hat = self.fc(out)
        return x_hat.unsqueeze(0) if len(x_hat.shape) == 1 else x_hat, y

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x_hat, y = self.prepare_batch(batch)  # logit
        y_hat = torch.sigmoid(x_hat)  # prob
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(y_hat, y)
        auc = self.auc(y_hat, y)
        f1 = self.f1(y_hat, y)
        ap = self.ap(y_hat, y.long())

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        self.log(
            "train_acc",
            accuracy,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        self.log(
            "train_auc",
            auc,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        self.log(
            "train_f1",
            f1,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        self.log(
            "train_ap",
            ap,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=len(y),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat, y = self.prepare_batch(batch)
        y_hat = torch.sigmoid(x_hat)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(y_hat, y)
        auc = self.auc(y_hat, y)
        f1 = self.f1(y_hat, y)
        ap = self.ap(y_hat, y.long())
        self.log("val_loss", loss, prog_bar=True, batch_size=len(y))
        self.log("val_acc", accuracy, prog_bar=True, batch_size=len(y))
        self.log("val_auc", auc, prog_bar=True, batch_size=len(y))
        self.log("val_f1", f1, prog_bar=True, batch_size=len(y))
        self.log("val_ap", ap, prog_bar=True, batch_size=len(y))

    def predict_step(self, batch):
        x_hat, y = self.prepare_batch(batch)
        y_hat = torch.sigmoid(x_hat)
        return y_hat, y

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5)
        return [optimizer], [
            {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}
        ]

    def compute_class_weights(self, dataset):
        """
        Compute class weights for binary classification.

        Args:
            dataset: The dataset containing labels.

        Returns:
            torch.Tensor: Class weights for the positive class.
        """
        labels = []
        # print(dataset[0])
        for i in range(len(dataset)):  # Assuming dataset returns (data, label, ...)
            labels.append(
                dataset[i][1].item()
                if isinstance(dataset[i][1], torch.Tensor)
                else dataset[i][1]
            )

        total_samples = len(labels)
        positive_samples = sum(labels)
        negative_samples = total_samples - positive_samples

        # Compute weights
        pos_weight = negative_samples / positive_samples
        print(f"Adjusting positive class weight to: {round(pos_weight, 3)}")
        return torch.tensor(pos_weight, dtype=torch.float32)


class LitLSTM(L.LightningModule):
    """LSTM using time-series data only.

    Args:
        L (_type_): _description_
    """

    def __init__(
        self,
        ts_input_dim,
        lstm_embed_dim,
        target_size,
        lr=0.1,
        with_packed_sequences=False,
    ):
        super().__init__()
        self.embed_timeseries = LSTM(
            ts_input_dim,
            lstm_embed_dim,
            target_size,
            with_packed_sequences=with_packed_sequences,
        )
        self.fc = nn.Linear(lstm_embed_dim, target_size)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self.acc = torchmetrics.Accuracy(task="binary")
        self.with_packed_sequences = with_packed_sequences

    def prepare_batch(self, batch):
        if self.with_packed_sequences:
            _, y, d, l = batch  # static, dynamic, lengths, labels  # noqa: E741
            d = torch.nn.utils.rnn.pack_padded_sequence(
                d, l, batch_first=True, enforce_sorted=False
            )

        else:
            _, y, d = batch

        ts_embed = self.embed_timeseries(d)

        # unpack if using packed sequences
        if self.with_packed_sequences:
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                ts_embed, batch_first=True
            )

        # [:, -1] for hidden state at the last time step
        logits = self.fc(lstm_out[:, -1])
        x_hat = F.sigmoid(logits)
        return x_hat, y

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x_hat, y = self.prepare_batch(batch)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_hat, y = self.prepare_batch(batch)
        loss = self.criterion(x_hat, y)
        accuracy = self.acc(x_hat, y)
        self.log("val_loss", loss, prog_bar=True, batch_size=len(y))
        self.log("val_acc", accuracy, prog_bar=True, batch_size=len(y))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SaveLossesCallback(Callback):
    def __init__(self, log_dir="logs", save_every_n_epochs=5):
        """
        Callback to save train/validation losses to a CSV file every n epochs.

        Args:
            log_dir (str): Directory to save the logs.
            save_every_n_epochs (int): Interval (in epochs) to save the losses.
        """
        self.log_dir = log_dir
        self.save_every_n_epochs = save_every_n_epochs
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_file = os.path.join(self.log_dir, "losses.csv")

        # Initialize the CSV file with headers if it doesn't exist
        with open(self.csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
        """
        # Save losses every n epochs
        if (trainer.current_epoch + 1) % self.save_every_n_epochs == 0:
            train_loss = trainer.callback_metrics.get("train_loss", None)
            val_loss = trainer.callback_metrics.get("val_loss", None)

            # Append the losses to the CSV file
            with open(self.csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        trainer.current_epoch + 1,
                        train_loss.item() if train_loss is not None else None,
                        val_loss.item() if val_loss is not None else None,
                    ]
                )

            print(f"Saved losses to {self.csv_file}")
