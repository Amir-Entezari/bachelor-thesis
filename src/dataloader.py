import os

import torch.nn.functional as F

import torch

import pandas as pd

from scipy.signal import resample

from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

class EEGDataLoader(torch.utils.data.Dataset):
    def __init__(self, dir, list_IDs):
        """
        Args:
            dir (str): Directory where the processed EEG chunks are stored.
            list_IDs (list): List of subject paths relative to the base directory.
            sampling_rate (int): Target sampling rate for resampling the data.
        """
        self.dir = dir
        self.list_IDs = list_IDs  # List of subject directories
        self.label_map = {
            "A": 0,
            "C": 1,
            "F": 2
        }
        # Gather all chunk file paths from the subject directories
        self.chunk_paths = []
        for subject_id in list_IDs:
            subject_dir = os.path.join(dir, subject_id)
            self.chunk_paths.extend(
                [
                    os.path.join(subject_id, file)
                    for file in os.listdir(subject_dir)
                    if file.endswith(".pt")
                ]
            )
        self.chunk_paths.sort()


    def __len__(self):
        """Returns the total number of chunks available."""
        return len(self.chunk_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index to retrieve a chunk and its label.

        Returns:
            X (torch.Tensor): EEG data as a tensor.
            y (int): Label corresponding to the chunk.
        """
        # Load the .pt file for the given chunk
        chunk_path = os.path.join(self.dir, self.chunk_paths[idx])
        sample = torch.load(chunk_path)
        # Extract data and labels
        eeg_data = sample["data"]
        label = sample["label"]

        # Extract the target label (e.g., Group)
        # X = eeg_data
        X = eeg_data.detach() if eeg_data.requires_grad else eeg_data
        y = label  # Adjust this if the target label changes
        return X.squeeze(), y


def collate_fn(batch):

    samples, labels = [], []

    for i, l in batch:

        samples.append(i)

        labels.append(l)

    samples = torch.stack(samples, dim = 0)

    labels = torch.tensor(labels)

    batch = samples

    return batch, labels



if __name__ == "__main__":
    print("HI")
    # Example usage
    # Path to dataset and labels
    data_dir = "dataset/derivatives/embeddings"
    participants_df = pd.read_csv('dataset/participants.tsv', sep='\t')
    subject_ids = participants_df['participant_id'].tolist()
    dataset = EEGDataLoader(data_dir, subject_ids)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(f"Batch Shape: {batch[0].shape}")
        break
