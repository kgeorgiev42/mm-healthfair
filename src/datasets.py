import argparse

import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from utils.functions import load_pickle, preview_data


class CollateFn:
    """Custom collate function for static data and labels."""

    def __call__(self, batch):
        static = torch.stack([data[0] for data in batch])
        labels = torch.stack([data[1] for data in batch])

        return static, labels


class CollateTimeSeries:
    """Custom collate function that can handle variable-length timeseries in a batch."""

    def __init__(self, method="pack_pad", min_events=None) -> None:
        self.method = method
        self.min_events = min_events

    def __call__(self, batch):
        static = torch.stack([data[0] for data in batch])
        labels = torch.stack([data[1] for data in batch])
        notes = None
        if len(batch[0]) > 3:  # noqa: PLR2004
            # pad notes to max length in batch
            notes = torch.stack([data[3] for data in batch])
            #notes = pad_sequence([data[3] for data in batch], batch_first=True)

        # number of dynamic timeseries data (note: dynamic is a list of timeseries)
        n_ts = len(batch[0][2])
        #print("Number of timeseries", n_ts)

        if self.method == "pack_pad":
            dynamic = []
            lengths = []
            for ts in range(n_ts):
                # Function to pad batch-wise due to timeseries of different lengths
                timeseries_lengths = [data[2][ts].shape[0] for data in batch]
                #print("Timeseries lengths", timeseries_lengths)
                max_events = max(timeseries_lengths)
                #print("Max events", max_events)
                n_ftrs = batch[0][2][ts].shape[1]
                events = torch.zeros((len(batch), max_events, n_ftrs))
                for i in range(len(batch)):
                    j, k = batch[i][2][ts].shape[0], batch[i][2][ts].shape[1]
                    events[i] = torch.concat(
                        [batch[i][2][ts], torch.zeros((max_events - j, k))]
                    )
                dynamic.append(events)
                lengths.append(timeseries_lengths)

            if notes is not None:
                return static, labels, dynamic, lengths, notes
            else:
                return static, labels, dynamic, lengths

        elif self.method == "truncate":
            # Truncate to minimum num of events in batch/ specified args

            dynamic = []
            n_ts = len(batch[0][2])
            for ts in range(n_ts):
                min_events = (
                    min([data[2][ts].shape[0] for data in batch])
                    if self.min_events is None
                    else self.min_events
                )
                events = [data[2][ts][:min_events] for data in batch]
                dynamic.append(events)

            if notes is not None:
                return static, labels, dynamic, lengths, notes
            else:
                return static, labels, dynamic, lengths


class MIMIC4Dataset(Dataset):
    """MIMIC-IV Dataset class. Subclass of Pytorch Dataset.
    Reads from .pkl data dictionary where key is patient ID and values are the dataframes.
    """

    def __init__(
        self,
        data_path=None,
        col_path=None,
        split=None,
        ids=None,
        static_only=False,
        with_notes=False,
        outcome="in_hosp_death"
    ) -> None:
        super().__init__()

        self.data_dict = load_pickle(data_path)
        self.col_dict = load_pickle(col_path)
        self.id_list = list(self.data_dict.keys()) if ids is None else ids
        self.dynamic_keys = sorted([key for key in self.data_dict[self.id_list[0]].keys() if "dynamic" in key])
        self.split = split
        self.static_only = static_only
        self.with_notes = with_notes
        self.splits = {"train": None, "val": None, "test": None}
        self.outcome = outcome
        self.splits[split] = ids

    def __len__(self):
        return (
            len(self.splits[self.split])
            if self.split is not None
            else len(self.id_list)
        )

    def __getitem__(self, idx):
        pt_id = int(self.splits[self.split][idx])
        static = self.data_dict[pt_id]["static"]
        label = torch.tensor(
            self.data_dict[pt_id][self.outcome][0][0], dtype=torch.float32
        ).unsqueeze(-1)
        static = torch.tensor(static, dtype=torch.float32)

        if self.static_only:
            return static, label

        else:
            dynamic = [
                self.data_dict[pt_id][i] for i in self.dynamic_keys
            ]
            dynamic = [torch.tensor(x, dtype=torch.float32) for x in dynamic]
            if self.with_notes:
                notes = self.data_dict[pt_id]["notes"]  # 1 x 768
                ### Extract tokens only
                emblist = []
                for emb in notes:
                    emblist.append(emb[1])

                notes = torch.tensor(emblist, dtype=torch.float32).unsqueeze(0)
                notes = torch.nn.functional.pad(notes, (0, 768 - notes.shape[1]))
                return static, label, dynamic, notes
            else:
                return static, label, dynamic

    def print_label_dist(self):
        # if no particular split then use entire data dict
        if self.split is None:
            id_list = self.id_list
        else:
            id_list = self.splits[self.split]

        n_positive = len([id_list[i] for i in range(len(id_list)) if self.data_dict[id_list[i]][self.outcome][0][0] == 1])

        if self.split is not None:
            print(f"{self.split.upper()}:")

        print(f"Positive cases: {n_positive}")
        print(
            f"Negative cases: {self.id_list.shape[0] - n_positive}"
        )

    def get_feature_dim(self, key="static"):
        return self.data_dict[int(self.id_list[0])][key].shape[1]

    def get_feature_list(self, key="static"):
        return self.col_dict[key + "_cols"]

    def get_split_ids(self, split):
        return self.splits[split]

if __name__ == "__main__":
    # Preview data from a saved pkl file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "processed_data_path",
        type=str,
        help="Path to the pickled data.",
    )
    args = parser.parse_args()
    preview_data(args.processed_data_path)
