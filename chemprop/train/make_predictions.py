from argparse import Namespace
import csv
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .predict import predict, get_emb
from chemprop.data import MoleculeDataset
from chemprop.models import build_model, build_pretrain_model
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
import pdb


def make_predictions(args: Namespace, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)

    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    if args.use_compound_names:
        compound_names = test_data.compound_names()
    print(f'Test size = {len(test_data):,}')

    # Normalize features
    if train_args.features_scaling:
        test_data.normalize_features(features_scaler)

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), args.num_tasks))
    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        model = load_checkpoint(checkpoint_path, cuda=args.cuda)
        
        # model = build_model(args, encoder_name=args.encoder_name)
        # model.encoder.load_state_dict(torch.load(args.checkpoint_path))
        
        # model_preds = predict(
        #     model=model,
        #     prompt=False,
        #     data=test_data,
        #     batch_size=args.batch_size,
        #     scaler=scaler
        # )
        model_preds = get_emb(
            model=model,
            prompt=False,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        # sum_preds += np.array(model_preds)

    # Ensemble predictions
    # avg_preds = sum_preds / len(args.checkpoint_paths)
    # avg_preds = avg_preds.tolist()
    # return avg_preds, test_data.smiles()
    return model_preds, test_data.smiles()


def get_embs(args: Namespace, smiles: List[str] = None) -> List[Optional[List[float]]]:
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :param smiles: Smiles to make predictions on.
    :return: A list of lists of target predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading data')
    if smiles is not None:
        test_data = get_data_from_smiles(smiles=smiles, skip_invalid_smiles=False)
    else:
        test_data = get_data(path=args.test_path, args=args, use_compound_names=args.use_compound_names, skip_invalid_smiles=False)
    pdb.set_trace()
    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    # Edge case if empty list of smiles is provided
    if len(test_data) == 0:
        return [None] * len(full_data)

    if args.use_compound_names:
        compound_names = test_data.compound_names()
    print(f'Test size = {len(test_data):,}')


    if args.checkpoint_path is not None:
        print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
        for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
            # Load model
            # model = load_checkpoint(checkpoint_path, cuda=args.cuda)
            model = build_pretrain_model(args, encoder_name=args.encoder_name)
            model.encoder.load_state_dict(torch.load(checkpoint_path))
            
            model = model.cuda()
            
            model_preds = get_emb(
                model=model,
                prompt=False,
                data=test_data,
                batch_size=args.batch_size,
                scaler=None
            )
    else:
        model = build_pretrain_model(args, encoder_name=args.encoder_name)
        model = model.cuda()

        model_preds = get_emb(
            model=model,
            prompt=False,
            data=test_data,
            batch_size=args.batch_size,
            scaler=None
        )

    return model_preds, test_data.smiles()