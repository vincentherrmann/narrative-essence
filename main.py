#!/usr/bin/env python3
# -*- coding: ascii -*-

from typing import Any, Callable, Dict, List
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import socket
from copy import deepcopy
from types import SimpleNamespace
from experiment_utilities.misc import fix_seed, seed_everything
from experiment_utilities.wandb_logging import Logger

from encoders import (LSTMAudioFeatureEncoder, PrecomputedAudioFeatureEncoder, PrecomputedAudioFeaturePCAEncoder,
                      OrderingLSTMEncoder)
from datasets import AudioFeatureDataset, AudioFeatureDatasetEchonest, collate_album_features_to_packed_seqs

__version__ = "1.0.0"


def parse_args(args: List[str] = sys.argv[1:]) -> Dict[str, Any]:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Trains a feature extractor computing narrative essence and a mutual information estimator "
                    "on the FMA dataset. Alternatively, the narrative essence extractor can be replaced by a"
                    "pre-computed feature.",
        prog="nee",
    )
    parser.add_argument('--wandb_logging', default=1, type=int, required=False,
                        help="Whether to log to wandb")
    parser.add_argument('--cuda', default=1, type=int, required=False,
                        help="Whether to use CUDA")
    parser.add_argument('--seed', default=-1, type=int, required=False,
                        help="Seed for the random number generator")
    parser.add_argument('--save_model', default=1, type=int, required=False,
                        help="Save snapshots of the models")
    parser.add_argument("--model_path", default="wandb_dir", type=str, required=False,
                        help="Path to save the model")
    parser.add_argument('--small_dataset', default=0, type=int, required=False,
                        help="Use the subset of data that includes echonest features (has to be 1 if any feature but "
                             "narrative essence is used)")
    parser.add_argument('--normalize_features', default=1, type=int, required=False,
                        help="Normalize the features computed by the feature extractor across a music album")
    parser.add_argument('--include_learned_feature', default=0, type=int, required=False,
                        help="whether to include the learned feature in the dataset set")
    parser.add_argument('--song_features', default="learned", type=str, required=False,
                        help="The features to use for the song. Options: "
                             "acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, "
                             "valence, duration, precomputed_learned, learned_frozen, learned, pca, all")

    parser.add_argument('--num_negative_examples', default=31, type=int, required=False,
                        help="Number of negative examples to use for the MI estimator")
    parser.add_argument('--patience_epochs', default=1, type=int, required=False,
                        help="Number of epochs to wait before early stopping")
    parser.add_argument('--feature_encoder_type', default="lstm", type=str, required=False,
                        help="Type of the feature encoder. Options: lstm, mean")
    parser.add_argument('--feature_encoder_hidden_units', default=128, type=int, required=False,
                        help="Number of hidden units in the feature encoder")
    parser.add_argument('--feature_encoder_num_layers', default=2, type=int, required=False,
                        help="Number of layers in the feature encoder")
    parser.add_argument('--feature_encoder_dropout', default=0.1, type=float, required=False,
                        help="Dropout in the feature encoder")
    parser.add_argument('--num_encoding_features', default=1, type=int, required=False,
                        help="Number of features to encode")

    parser.add_argument('--ordering_encoder_hidden_units', default=32, type=int, required=False,
                        help="Number of hidden units in the ordering encoder")
    parser.add_argument('--ordering_encoder_num_layers', default=2, type=int, required=False,
                        help="Number of layers in the ordering encoder")
    parser.add_argument('--ordering_encoder_bidirectional', default=1, type=int, required=False,
                        help="Whether the ordering encoder is bidirectional")
    parser.add_argument('--ordering_encoder_dropout', default=0.0, type=float, required=False,
                        help="Dropout in the ordering encoder")
    parser.add_argument('--ordering_encoder_weight_decay', default=1e-5, type=float, required=False,
                        help="Weight decay in the ordering encoder")

    return parser.parse_args(args)


def train_model(args):
    log_with_wandb = args.wandb_logging > 0
    args.server = socket.gethostname()
    hyperparameters = dict(vars(args))

    logger = Logger(enabled=log_with_wandb,
                    print_logs_to_console=not log_with_wandb,
                    project="narrative-essence",
                    tags=[],
                    config=hyperparameters)

    if args.model_path == "wandb_dir" and log_with_wandb:
        model_path = logger().run.dir
    else:
        model_path = args.model_path

    dev = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu") if args.cuda else 'cpu'
    #dev = 'cpu'

    args.save_model = args.save_model and not args.small_dataset and args.song_features == "learned"

    if args.seed >= 0:
        seed_everything(args.seed)
    num_negative_examples = args.num_negative_examples

    use_learned_feature = args.include_learned_feature

    dataset_class = (
        AudioFeatureDatasetEchonest if args.small_dataset else AudioFeatureDataset
    )

    training_set = dataset_class(
        mode="training", include_learned_feature=use_learned_feature
    )
    training_loader = torch.utils.data.DataLoader(
        dataset=training_set,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_album_features_to_packed_seqs,
    )

    validation_set = dataset_class(
        mode="validation", include_learned_feature=use_learned_feature
    )
    if not args.small_dataset:
        small_validation_set = AudioFeatureDatasetEchonest(
            mode="validation", include_learned_feature=use_learned_feature
        )

    if args.song_features == "learned" or args.song_features == "learned_frozen":
        feature_encoder = LSTMAudioFeatureEncoder(
            num_in_features=7,
            hidden_size=args.feature_encoder_hidden_units,
            num_layers=args.feature_encoder_num_layers,
            bidirectional=True,
            dropout=args.feature_encoder_dropout if args.song_features == "learned" else 0.0,
            num_out_features=args.num_encoding_features,
        )
        num_encoding_features = args.num_encoding_features
        if args.song_features == "learned_frozen":

            saved_encoder = torch.load(os.path.join(model_path, 'feature_encoder.p'))
            feature_encoder.load_state_dict(saved_encoder.state_dict())
            feature_encoder.eval()
            for param in feature_encoder.parameters():
                param.requires_grad = False
    elif args.song_features == "pca":
        num_encoding_features = args.num_encoding_features
        feature_encoder = PrecomputedAudioFeaturePCAEncoder(
            num_output_features=num_encoding_features,
            dataset=training_set
        )
    else:
        feature_encoder = PrecomputedAudioFeatureEncoder(args.song_features)
        if args.song_features == "all":
            num_encoding_features = 9 if use_learned_feature else 8
        else:
            num_encoding_features = 1

    feature_encoder = feature_encoder.to(dev)

    ordering_encoder = OrderingLSTMEncoder(
        input_size=num_encoding_features,
        hidden_size=args.ordering_encoder_hidden_units,
        num_layers=args.ordering_encoder_num_layers,
        bidirectional=args.ordering_encoder_bidirectional,
        dropout=args.ordering_encoder_dropout,
    )
    ordering_encoder = ordering_encoder.to(dev)

    feature_encoder_optimizer = torch.optim.Adam(feature_encoder.parameters(), lr=1e-4)
    ordering_encoder_optimizer = torch.optim.Adam(
        ordering_encoder.parameters(),
        lr=1e-4,
        weight_decay=args.ordering_encoder_weight_decay,
    )

    @fix_seed
    def validate(validation_set, num_iterations=5):
        feature_encoder.eval()
        ordering_encoder.eval()
        all_losses = []
        num_tracks = range(3, 21)
        losses_per_num_tracks = {i: [] for i in num_tracks}

        validation_loader = torch.utils.data.DataLoader(
            dataset=validation_set,
            batch_size=16,
            shuffle=False,
            collate_fn=collate_album_features_to_packed_seqs,
        )

        with torch.no_grad():
            for iteration in range(num_iterations):
                for batch in iter(validation_loader):
                    batch = {k: v.to(dev) for k, v in batch.items()}
                    seq_lengths = batch["sequence_lengths"]
                    batch_size = seq_lengths.shape[0]

                    padded_features_dict = {
                        k: v for k, v in batch.items() if k != "sequence_lengths"
                    }
                    padded_features_dict = {
                        k: torch.nn.utils.rnn.pad_packed_sequence(v, padding_value=-1)[0] for k, v in
                        padded_features_dict.items()
                    }

                    narrative_features = feature_encoder(padded_features_dict)
                    narrative_features = narrative_features.view(
                        -1, batch_size, num_encoding_features
                    )

                    feature_mask = torch.arange(narrative_features.shape[0], device=dev).unsqueeze(
                        1
                    ) < seq_lengths.unsqueeze(0)
                    feature_mask = feature_mask.unsqueeze(2).float().to(dev)
                    valid_features = narrative_features * feature_mask

                    # normalize features
                    feature_mean = valid_features.sum(dim=0) / feature_mask.sum(dim=0)
                    feature_var = ((valid_features - feature_mean.unsqueeze(0)) ** 2).sum(
                        dim=0
                    ) / feature_mask.sum(dim=0)
                    feature_std = feature_var ** 0.5
                    valid_features = (valid_features - feature_mean) / feature_std

                    narrative_features = valid_features

                    padded_track_number = padded_features_dict["track_numbers"]
                    r_batch = (
                        torch.arange(batch_size)
                        .unsqueeze(1)
                        .repeat(1, num_negative_examples + 1)
                        .flatten()
                    )
                    r_seq_lengths = (
                        seq_lengths.unsqueeze(1).repeat(1, num_negative_examples + 1).flatten()
                    )
                    permutation = [
                        torch.arange(l) if i % batch_size == 0 else torch.randperm(l)
                        for i, l in enumerate(r_seq_lengths)
                    ]
                    padded_permutations = torch.nn.utils.rnn.pad_sequence(permutation).to(dev)
                    shuffled_narrative_features = narrative_features[
                                                  padded_permutations, r_batch.unsqueeze(0), :
                                                  ]

                    ordering_scores, ordering_encodings = ordering_encoder(
                        shuffled_narrative_features, r_seq_lengths
                    )
                    ordering_scores = ordering_scores.view(
                        batch_size, num_negative_examples + 1
                    )
                    targets = torch.zeros(batch_size, dtype=torch.long).to(dev) + 0

                    loss = F.cross_entropy(ordering_scores.detach(), targets, reduction="none")
                    for i, l in enumerate(loss):
                        losses_per_num_tracks[seq_lengths[i].item()].append(l.cpu().detach())
                    all_losses.append(loss.detach())
        mean_loss_per_num_tracks = {
            k: torch.stack(v).mean().item() if len(v) > 0 else 0.0
            for k, v in losses_per_num_tracks.items()
        }
        all_losses = torch.cat(all_losses)
        validation_loss = all_losses.mean()
        feature_encoder.train()
        ordering_encoder.train()
        return validation_loss.item(), mean_loss_per_num_tracks

    step = 0
    lowest_validation_loss = float("inf")
    best_epoch = 0
    for epoch in range(200):
        print(f"epoch {epoch}")
        num_tracks = range(3, 21)
        losses_per_num_tracks = {i: [] for i in num_tracks}
        for batch in iter(training_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            seq_lengths = batch["sequence_lengths"]
            batch_size = seq_lengths.shape[0]

            padded_features_dict = {
                k: v for k, v in batch.items() if k != "sequence_lengths"
            }
            padded_features_dict = {
                k: torch.nn.utils.rnn.pad_packed_sequence(v, padding_value=-1)[0] for k,v in padded_features_dict.items()
            }

            narrative_features = feature_encoder(padded_features_dict)
            narrative_features = narrative_features.view(
                -1, batch_size, num_encoding_features
            )  # padded_sequence_length x batch_size x num_encoding_features

            feature_mask = torch.arange(narrative_features.shape[0], device=dev).unsqueeze(
                1
            ) < seq_lengths.unsqueeze(0)
            feature_mask = feature_mask.unsqueeze(2).float().to(dev)
            valid_features = narrative_features * feature_mask

            # normalize features
            feature_mean = valid_features.sum(dim=0) / feature_mask.sum(dim=0)
            feature_var = ((valid_features - feature_mean.unsqueeze(0)) ** 2).sum(
                dim=0
            ) / feature_mask.sum(dim=0)
            feature_std = feature_var ** 0.5
            valid_features = (valid_features - feature_mean) / feature_std

            narrative_features = valid_features

            padded_track_number = padded_features_dict["track_numbers"]
            r_batch = (
                torch.arange(batch_size)
                .unsqueeze(1)
                .repeat(1, num_negative_examples + 1)
                .flatten()
            )
            r_seq_lengths = (
                seq_lengths.unsqueeze(1).repeat(1, num_negative_examples + 1).flatten()
            )
            permutation = [
                torch.arange(l) if i % batch_size == 0 else torch.randperm(l)
                for i, l in enumerate(r_seq_lengths)
            ]
            padded_permutations = torch.nn.utils.rnn.pad_sequence(permutation).to(dev)
            shuffled_narrative_features = narrative_features[
                                          padded_permutations, r_batch.unsqueeze(0), :
                                          ]

            ordering_scores, ordering_encodings = ordering_encoder(
                shuffled_narrative_features, r_seq_lengths
            )
            ordering_scores = ordering_scores.view(
                batch_size, num_negative_examples + 1
            )
            targets = torch.zeros(batch_size, dtype=torch.long).to(dev) + 0

            losses = F.cross_entropy(ordering_scores, targets, reduction="none")
            loss = losses.mean()
            feature_encoder_optimizer.zero_grad()
            ordering_encoder_optimizer.zero_grad()
            loss.backward()
            if step % 100 == 0:
                logger().log({
                    "training_loss": loss.item(),
                }, step=step)
            feature_encoder_optimizer.step()
            ordering_encoder_optimizer.step()

            for i, l in enumerate(losses):
                losses_per_num_tracks[seq_lengths[i].item()].append(l.cpu().detach())

            step += 1
        validation_loss, validation_mean_loss_per_num_tracks = validate(validation_set)
        validation_mi_lower_bound = np.log(num_negative_examples + 1) - validation_loss

        training_mean_loss_per_num_tracks = {
            k: torch.stack(v).mean().item() if len(v) > 0 else 0.0
            for k, v in losses_per_num_tracks.items()
        }

        if validation_loss < lowest_validation_loss:
            lowest_validation_loss = validation_loss
            best_epoch = epoch

            if args.save_model:
                print("save models")
                torch.save(
                    feature_encoder.cpu(),
                    os.path.join(model_path, "feature_encoder.p"),
                )
                torch.save(
                    ordering_encoder.cpu(),
                    os.path.join(model_path, "ordering_encoder.p"),
                )
                feature_encoder.to(dev)
                ordering_encoder.to(dev)
        else:
            if epoch - best_epoch > args.patience_epochs:
                print(f"early stopping?best model after epoch {best_epoch}")
                break

        validation_mi_lower_bound_bits = validation_mi_lower_bound * np.log2(np.e)
        logger().log({
            "validation_loss": validation_loss,
            "validation_mi_lower_bound": validation_mi_lower_bound_bits,
            "epoch": epoch,
        }, step=step)

    highest_mi_lower_bound = np.log(num_negative_examples + 1) - lowest_validation_loss
    highest_mi_lower_bound_bits = highest_mi_lower_bound * np.log2(np.e)
    logger().log({
        "highest_mi_lower_bound": highest_mi_lower_bound_bits,
    }, step=step)

    if args.song_features == "learned" and args.save_model:
        eval_args = deepcopy(args)
        eval_args.wandb_logging = 0
        eval_args.save_model = 0
        eval_args.model_path = model_path
        eval_args.song_features = "learned_frozen"
        eval_args.small_dataset = 1
        eval_args.patience_epochs = 100

        echonest_mi = train_model(eval_args)
        logger().log({
            "echonest_highest_mi_lower_bound": echonest_mi,
        }, step=step)

    return highest_mi_lower_bound_bits


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
