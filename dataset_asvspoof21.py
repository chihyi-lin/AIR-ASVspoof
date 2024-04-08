
import argparse
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from src.datasets.detection_dataset import DetectionDataset
from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset
import pandas as pd

from torch.utils.data import DataLoader


def get_datasets(
    datasets_paths: List[Union[Path, str]],
    amount_to_use: Tuple[Optional[int], Optional[int]],
) -> Tuple[DetectionDataset, DetectionDataset]:
    data_train = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        subset="train",
        reduced_number=amount_to_use[0],
        oversample=True,
    )
    data_test = DetectionDataset(
        asvspoof_path=datasets_paths[0],
        subset="test",
        reduced_number=amount_to_use[1],
        oversample=True,
    )

    return data_train, data_test


# def parse_args():
#     parser = argparse.ArgumentParser()

#     ASVSPOOF_DATASET_PATH = "datasets/ASVspoof2021/DF"

#     parser.add_argument(
#         "--asv_path",
#         type=str,
#         default=ASVSPOOF_DATASET_PATH,
#         help="Path to ASVspoof2021 dataset directory",
#     )
    
#     default_train_amount = None
#     parser.add_argument(
#         "--train_amount",
#         "-a",
#         help=f"Amount of files to load for training.",
#         type=int,
#         default=default_train_amount,
#     )

#     default_val_amount = None
#     parser.add_argument(
#         "--valid_amount",
#         "-va",
#         help=f"Amount of files to load for testing.",
#         type=int,
#         default=default_val_amount,
#     )

#     return parser.parse_args()


def preprocess_data(
        datasets_paths,
        train_amount,
        valid_amount,
        batch_size
):
    # get_datasets, transforming waveforms into feature vectors
    data_train, data_test = get_datasets(
        datasets_paths=[datasets_paths],
        amount_to_use=(train_amount, valid_amount),
    )
    print(f"Using mfcc_double_delta as features!")

    train_loader = DataLoader(
            data_train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
    
    test_loader = DataLoader(
            data_test,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=6,
        )
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = preprocess_data()