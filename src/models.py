import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
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
        dropout=0.1,
        num_ts=2,
        target_size=1,
        lr=0.1,
        fusion_method="concat",
        st_first=True,
        with_ts=False,
        with_notes=False,
        with_packed_sequences=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_ts = num_ts
        self.with_ts = with_ts
        self.with_notes = with_notes
        self.st_first = st_first
        self.fusion_method = fusion_method

        self.embed_static = nn.Sequential(
            nn.Linear(st_input_dim, st_embed_dim // 2),
            nn.LayerNorm(st_embed_dim // 2),
            nn.Linear(st_embed_dim // 2, st_embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
        )

        if self.with_ts:
            self.embed_timeseries = nn.ModuleList(
                [
                    LSTM(
                        ts_input_dim[i],
                        ts_embed_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                    )
                    for i in range(self.num_ts)
                ]
            )

        if self.with_notes:
            self.embed_notes = nn.Linear(nt_input_dim, nt_embed_dim)
        else:
            self.embed_notes = None
            nt_embed_dim = 0

        if self.fusion_method == "mag":
            if self.st_first:
                self.fuse = Gate(
                    st_embed_dim, *([ts_embed_dim] * self.num_ts), dropout=dropout
                )
                self.fc = nn.Linear(st_embed_dim, target_size)

            else:
                self.fuse = Gate(
                    *([ts_embed_dim] * self.num_ts), st_embed_dim, dropout=dropout
                )
                self.fc = nn.Linear(ts_embed_dim, target_size)

        elif self.fusion_method == "concat":
            # embeddings must be same dim
            assert st_embed_dim == ts_embed_dim
            if self.with_notes:
                assert nt_embed_dim == st_embed_dim
            self.fc = nn.Linear(
                st_embed_dim + (self.num_ts * ts_embed_dim) + nt_embed_dim, target_size
            )

        elif self.fusion_method == "None":
            self.fc = nn.Linear(st_embed_dim, target_size)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.lr = lr
        self.acc = torchmetrics.Accuracy(task="binary")
        self.auc = torchmetrics.AUROC(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.ap = torchmetrics.AveragePrecision(task="binary")

        self.with_packed_sequences = with_packed_sequences

    def prepare_batch(self, batch):  # noqa: PLR0912
        # static, labels, dynamic, lengths, notes (optional) # noqa: E741
        s = batch[0]
        y = batch[1]

        if self.with_ts:
            d = batch[2]
            if self.with_packed_sequences:
                lengths = batch[3]

        if self.fusion_method != "None":
            # print('Packing padded sequences.')
            ts_embed = []
            for i in range(self.num_ts):
                if self.with_packed_sequences:
                    packed_d = torch.nn.utils.rnn.pack_padded_sequence(
                        d[i], lengths[i], batch_first=True, enforce_sorted=False
                    )
                    embed = self.embed_timeseries[i](packed_d)
                else:
                    embed = self.embed_timeseries[i](d[i])

                ts_embed.append(embed.unsqueeze(1))

        if self.with_notes:
            n = batch[4]
            nt_embed = self.embed_notes(n)
        else:
            nt_embed = None

        st_embed = self.embed_static(s)

        # Fuse time-series and static data
        if self.fusion_method == "concat":
            # use * to allow variable number of ts_embeddings
            # concat along feature dim
            embeddings = [st_embed, *ts_embed]
            embeddings = embeddings + [nt_embed] if nt_embed is not None else embeddings
            out = torch.concat(embeddings, dim=-1).squeeze()  # b x dim*2
        elif self.fusion_method == "mag":
            if self.st_first:
                out = self.fuse(st_embed, *ts_embed)  # b x st_embed_dim
            else:
                out = self.fuse(*ts_embed, st_embed)

        elif self.fusion_method == "None":
            # print('No fusion method specified. Using static data only.')
            out = st_embed.squeeze()

        # Parse through FC
        x_hat = self.fc(out)  # b x 1 - logits
        if len(x_hat.shape) < 2:  # noqa: PLR2004
            x_hat = x_hat.unsqueeze(0)
        return x_hat, y

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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=20)
        return [optimizer], [
            {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch"}
        ]

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
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        return optimizer
