import sys

import os

import torch

import random

import numpy as np

import pandas as pd

import mne

from src import dataloader, models, train_multiclass

from torch.utils.data import DataLoader

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl

from src.dataloader import *
from src.models import *
from src.train_multiclass import *

# Check if exactly one argument is passed (besides the script name)
if len(sys.argv) != 2:
    print("Error: Too many argument.")
    sys.exit(1)

# Get the argument
split_index = sys.argv[1]

# Try to convert the argument to a number
try:
    split_index = int(split_index) 
    if 0 <= split_index < 88:
        print(f"The split index {split_index} is valid.")
    else:
        print("The split index is not between 0 and 88.")
        sys.exit(1)
except ValueError:
    print("The argument is not a valid split index.")
    sys.exit(1)



DATA_DIR = 'dataset'

participants_path = os.path.join(DATA_DIR, 'participants.tsv')
participants_df = pd.read_csv(participants_path, sep='\t')

subject_ids = participants_df['participant_id'].tolist()

# Perform LOSO cross-validation splits
loso_splits = []
for test_subject in subject_ids:
    # Test set is the current subject
    test_set = [test_subject]

    # Randomly select 6 subjects for validation
    validation_subjects = random.sample(subject_ids, 6)
    while test_subject in validation_subjects:
        validation_subjects = random.sample(subject_ids, 6)
        
    train_set = []
    # Training set is all other subjects
    for subject in subject_ids:
        if subject != test_subject and subject not in validation_subjects:# and int(subject[-3:]) <=3:
            train_set.append(subject)
    # train_set = [subject for subject in subject_ids if subject != test_subject]

    # Append the split to the list
    loso_splits.append({'train': train_set, 
                        'val': validation_subjects,
                        'test': test_set})

# for i, split in enumerate(loso_splits):
#     print(f"Fold {i + 1}:")
#     print(f"  Train: {split['train']}")
#     print(f"  Test: {split['test']}")
#     print()

print(f"LOSO split {split_index}, Train Subjects: ", loso_splits[split_index]['train'])
print(f"LOSO split {split_index}, Validation Subjects: ", loso_splits[split_index]['val'])
print(f"LOSO split {split_index}, Test Subjects: ", loso_splits[split_index]['test'])



def train_loso(trian_set,
                val_set, 
                test_set,
                device=[0],
                max_epochs=100
                ):
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Define root directories
    embeddings_dir = os.path.join(DATA_DIR, 'derivatives/embeddings')
    labels_file = os.path.join(DATA_DIR, 'participants.tsv')

    train_loader = DataLoader(
            EEGDataLoader(embeddings_dir, trian_set),
            batch_size=64,
            shuffle=True,
            drop_last=False,
            num_workers=2,
            persistent_workers=True,
            collate_fn = collate_fn
        )
    val_loader = DataLoader(
        EEGDataLoader(embeddings_dir, val_set),
        batch_size=64,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        persistent_workers=True,
        collate_fn = collate_fn
    )
    test_loader = DataLoader(
        EEGDataLoader(embeddings_dir, test_set),
        batch_size=64,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        persistent_workers=True,
        collate_fn = collate_fn
    )

    # define the model
    model_name = "Moment"
    if model_name == "Moment":
        model = MomentClassifier(512, n_classes = 3,n_channels=19)
    else:
        raise NotImiplementedError
    lightning_model = LitModel_finetune(model=model,args={
            "lr":0.01,
            "weight_decay":1e-5,
            "gamma":0.1,
            "n_step":100,
        })

    # logger and callbacks
    version = "moment"
    logfolder = "log"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name=logfolder,
    )
    early_stop_callback = EarlyStopping(
        monitor="accuracy", patience= 20, verbose=False, mode="max"
    )

    checkpoint_callback = ModelCheckpoint(save_top_k = 0,
                                        monitor = "epoch",
                                        mode = "max",
                                        save_last = True
                                            )

    tqdm_progress_bar = TQDMProgressBar(refresh_rate= 20, process_position=0)

    trainer = pl.Trainer(
        # devices=1,  # Set devices to an integer instead of a list when using CPU
        devices=device,  # Use list format only when specifying GPUs
        accelerator="auto",
        strategy=DDPStrategy(),
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=max_epochs,
        callbacks= [checkpoint_callback, tqdm_progress_bar], # [early_stop_callback, tqdm_progress_bar],
        log_every_n_steps = 1,
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="last", dataloaders=test_loader
    )[0]
    print(pretrain_result)
    return pretrain_result

pretrain_result=train_loso(loso_splits[split_index]['train'], loso_splits[split_index]['val'], loso_splits[split_index]['test'], device=[1], max_epochs=100)

# accuracy_list=[]
# f1_list=[]
# for i in range(88):
#     print(f"****************Test Subject: {i} ******************************",)
#     pretrain_result=train_loso(loso_splits[i]['train'], loso_splits[i]['val'], loso_splits[i]['test'], device=[1], max_epochs=1)
#     accuracy_list.append(pretrain_result['test_accuracy'])
#     f1_list.append(pretrain_result['test_f1_weighted'])

#     print("Accuracy List: ", accuracy_list)
#     print("Average Accuracy: ", sum(accuracy_list) / len(accuracy_list))

#     print("F1 List: ",f1_list)
#     print("Average F1: ", sum(f1_list) / len(f1_list))
