import torch
import os
import json
import urllib
from typing import Dict, Any
from pathlib import Path


class AudioFeatureDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            mode="train",
            allow_albums_with_missing_tracks=False,
            shuffle_track_orderings=False,
            sort_track_orderings=False,
            normalize_features=True,
            include_learned_feature=False,
    ):
        super().__init__()
        self.data_dir = os.path.join(Path(__file__).parent, "data")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.allow_albums_with_missing_tracks = allow_albums_with_missing_tracks
        self.include_learned_feature = include_learned_feature
        dataset = self.create_dataset_file(allow_albums_with_missing_tracks)

        self.audio_features_mean = dataset["audio_features"].mean(dim=0)
        self.audio_features_std = dataset["audio_features"].std(dim=0)
        self.audio_features_std[self.audio_features_std == 0.0] = 1.0

        self.durations_mean = dataset["durations"].float().mean(dim=0)
        self.durations_std = dataset["durations"].float().std(dim=0)

        if normalize_features:
            dataset["audio_features"] = (
                                                dataset["audio_features"] - self.audio_features_mean
                                        ) / self.audio_features_std
            dataset["durations"] = (
                                           dataset["durations"].float() - self.durations_mean
                                   ) / self.durations_std

        if shuffle_track_orderings:
            dataset = self.shuffle_indices(dataset)
        elif sort_track_orderings:
            dataset = self.sort_album_tracks_by_audio_feature_mean(dataset)

        album_indices = dataset["album_indices"]
        split = dataset["split"]

        if mode == "validation":
            split_mode = 1
        elif mode == "test":
            split_mode = 2
        else:
            split_mode = 0

        album_in_dataset = split[album_indices] == split_mode
        if mode == "full":
            album_in_dataset = torch.ones_like(album_in_dataset)
        self.album_indices = album_indices[album_in_dataset]
        self.album_lengths = dataset["album_lengths"][album_in_dataset]
        self.album_ids = dataset["album_ids"][album_in_dataset]

        self.audio_features = dataset["audio_features"]
        self.durations = dataset["durations"]
        self.track_numbers = dataset["track_numbers"]

    def __getitem__(self, item):
        album_idx = self.album_indices[item]
        l = self.album_lengths[item]
        a = self.audio_features[album_idx: album_idx + l]
        d = self.durations[album_idx: album_idx + l]
        n = self.track_numbers[album_idx: album_idx + l]

        features = torch.cat([a, d.unsqueeze(1).repeat(1, 7)], dim=1)

        if self.include_learned_feature:
            album_id = self.album_ids[item].item()
            learned_features = self.learned_feature_dict[album_id]
            return {
                "features": features,
                "track_numbers": n,
                "learned_features": learned_features.unsqueeze(1),
            }
        else:
            return {"features": features, "track_numbers": n}

    def __len__(self):
        return len(self.album_indices)

    def shuffle_indices(self, dataset):
        # shuffle the track ordering of each album. Only for testing purposes.
        print("shuffle track orderings")
        index_lookup = []
        for i in range(len(dataset["album_indices"])):
            album_idx = dataset["album_indices"][i]
            l = dataset["album_lengths"][i]
            p = torch.randperm(l)
            index_lookup.append(album_idx + p)
        index_lookup = torch.cat(index_lookup)
        dataset["audio_features"] = dataset["audio_features"][index_lookup]
        dataset["durations"] = dataset["durations"][index_lookup]
        dataset["track_numbers"] = dataset["track_numbers"][index_lookup]
        return dataset

    def sort_album_tracks_by_audio_feature_mean(self, dataset):
        # sort the track ordering of each album by the mean of the audio features, so that we can be sure that there is
        # some structure. Only for testing purposes.
        print("sort track orderings")
        audio_feature_mean = dataset["audio_features"].mean(dim=1)

        index_lookup = []
        for i in range(len(dataset["album_indices"])):
            album_idx = dataset["album_indices"][i]
            l = dataset["album_lengths"][i]
            sorting_criterion = audio_feature_mean[album_idx: album_idx + l]
            p = torch.sort(sorting_criterion)
            index_lookup.append(album_idx + p[1])
        index_lookup = torch.cat(index_lookup)
        audio_feature_mean_sorted = audio_feature_mean[index_lookup]
        dataset["audio_features"] = dataset["audio_features"][index_lookup]
        dataset["durations"] = dataset["durations"][index_lookup]
        dataset["track_numbers"] = dataset["track_numbers"][index_lookup]
        return dataset

    def create_dataset_file(self, allow_albums_with_missing_tracks: bool = True) -> Dict[str, Any]:
        if allow_albums_with_missing_tracks:
            self.dataset_file = os.path.join(
                self.data_dir, "fma_album_audio_feature_dataset.p"
            )
        else:
            self.dataset_file = os.path.join(
                self.data_dir, "fma_album_audio_feature_dataset_only_full_albums.p"
            )

        if self.include_learned_feature:
            self.learned_feature_dict = {}
            json_path = os.path.join(self.data_dir, "fma_albums_learned_feature.json")
            if not os.path.exists(json_path):
                print("downloading fma_albums_learned_feature.json")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/narrative-essence-public-data/fma_albums_learned_feature.json",
                    json_path,
                )
            with open(json_path, "rb") as f:
                albums_with_learned_features = json.load(f)
            for album in albums_with_learned_features:
                learned_features = torch.Tensor(
                    [t["learned scalar feature"] for t in album]
                )
                self.learned_feature_dict[album[0]["album id"]] = learned_features

        if os.path.exists(self.dataset_file):
            return torch.load(self.dataset_file)

        print("creating dataset file")
        json_path = os.path.join(self.data_dir, "fma_albums_full.json")

        if not os.path.exists(json_path):
            print("downloading fma_albums_full.json")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/narrative-essence-public-data/fma_albums_full.json",
                json_path,
            )

        with open(json_path, "rb") as f:
            dataset_list = json.load(f)

        filtered_dataset_list = []
        for album in dataset_list:
            track_numbers = [t["track number"] for t in album]
            if track_numbers[0] == 0:
                print("add 1 to track numbers")
                track_numbers = [n + 1 for n in track_numbers]
                for t in album:
                    t["track number"] += 1
            duplicates = [x for x in track_numbers if track_numbers.count(x) > 1]
            if len(duplicates) > 0:
                print("skip album")
                continue
            if not self.allow_albums_with_missing_tracks:
                num_album_tracks = album[0]["album tracks"]
                if len(track_numbers) != num_album_tracks:
                    print("skip album")
                    continue
                if track_numbers[-1] != num_album_tracks:
                    print("skip album")
                    continue
            filtered_dataset_list.append(album)

        audio_features = []
        durations = []
        track_numbers = []
        split = []
        album_indices = []
        album_lengths = []
        album_ids = []

        for album in filtered_dataset_list:
            album_indices.append(len(durations))
            album_lengths.append(len(album))
            for track in album:
                audio_features.append(torch.Tensor(track["fma_audio_features"]))
                durations.append(track["track duration"])
                track_numbers.append(track["track number"])
                split.append(track["set split"])
            album_ids.append(track["album id"])

        audio_features = torch.stack(audio_features, dim=0)
        durations = torch.LongTensor(durations)
        track_numbers = torch.LongTensor(track_numbers)
        split_dict = {"training": 0, "validation": 1, "test": 2}
        split = torch.Tensor([split_dict[s] for s in split]).long()
        album_indices = torch.LongTensor(album_indices)
        album_lengths = torch.LongTensor(album_lengths)
        album_ids = torch.LongTensor(album_ids)

        dataset_list = {
            "audio_features": audio_features,
            "durations": durations,
            "split": split,
            "track_numbers": track_numbers,
            "album_indices": album_indices,
            "album_lengths": album_lengths,
            "album_ids": album_ids,
        }

        torch.save(dataset_list, self.dataset_file)
        return torch.load(self.dataset_file)


class AudioFeatureDatasetEchonest(AudioFeatureDataset):
    def __init__(
            self,
            mode="train",
            allow_albums_with_missing_tracks=False,
            shuffle_track_orderings=False,
            sort_track_orderings=False,
            normalize_features=True,
            include_learned_feature=False,
    ):

        super().__init__(
            mode=mode,
            allow_albums_with_missing_tracks=allow_albums_with_missing_tracks,
            shuffle_track_orderings=shuffle_track_orderings,
            sort_track_orderings=sort_track_orderings,
            normalize_features=normalize_features,
            include_learned_feature=include_learned_feature,
        )

        dataset = torch.load(self.dataset_file)

        self.echonest_features_mean = dataset["echonest_features"].mean(dim=0)
        self.echonest_features_std = dataset["echonest_features"].std(dim=0)
        self.echonest_features_std[self.echonest_features_std == 0.0] = 1.0

        if normalize_features:
            dataset["echonest_features"] = (
                                                   dataset["echonest_features"] - self.echonest_features_mean
                                           ) / self.echonest_features_std

        self.echonest_features = dataset["echonest_features"]

    def __getitem__(self, item):
        album_idx = self.album_indices[item]
        l = self.album_lengths[item]
        a = self.audio_features[album_idx: album_idx + l]
        d = self.durations[album_idx: album_idx + l]
        n = self.track_numbers[album_idx: album_idx + l]
        e = self.echonest_features[album_idx: album_idx + l]

        features = torch.cat([a, d.unsqueeze(1).repeat(1, 7)], dim=1)
        #features = torch.cat([e, d.unsqueeze(1)], dim=1)

        if self.include_learned_feature:
            album_id = self.album_ids[item].item()
            learned_features = self.learned_feature_dict[album_id]
            return {
                "features": features,
                "echonest_features": e,
                "track_numbers": n,
                "learned_features": learned_features.unsqueeze(1),
                "duration": d
            }
        else:
            return {
                "features": features,
                "echonest_features": e,
                "track_numbers": n,
                "duration": d
            }

    def create_dataset_file(self, allow_albums_with_missing_tracks=True):
        if allow_albums_with_missing_tracks:
            self.dataset_file = os.path.join(
                self.data_dir, "fma_album_echonest_audio_feature_dataset.p"
            )
        else:
            self.dataset_file = os.path.join(
                self.data_dir,
                "fma_album_echonest_audio_feature_dataset_only_full_albums.p",
            )

        if self.include_learned_feature:
            self.learned_feature_dict = {}
            json_path = os.path.join(self.data_dir, "fma_albums_learned_feature.json")
            if not os.path.exists(json_path):
                print("downloading fma_albums_learned_feature.json")
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/narrative-essence-public-data/fma_albums_learned_feature.json",
                    json_path,
                )
            with open(json_path, "rb") as f:
                albums_with_learned_features = json.load(f)
            for album in albums_with_learned_features:
                learned_features = torch.Tensor(
                    [t["learned scalar feature"] for t in album]
                )
                self.learned_feature_dict[album[0]["album id"]] = learned_features

        if os.path.exists(self.dataset_file):
            return torch.load(self.dataset_file)

        print("creating dataset file")
        json_path = os.path.join(self.data_dir, "fma_albums_small.json")

        if not os.path.exists(json_path):
            print("downloading fma_albums_small.json")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/narrative-essence-public-data/fma_albums_small.json",
                json_path,
            )

        with open(json_path, "rb") as f:
            dataset_list = json.load(f)

        filtered_dataset_list = []
        for album in dataset_list:
            track_numbers = [t["track number"] for t in album]
            if track_numbers[0] == 0:
                print("add 1 to track numbers")
                track_numbers = [n + 1 for n in track_numbers]
                for t in album:
                    t["track number"] += 1
            duplicates = [x for x in track_numbers if track_numbers.count(x) > 1]
            if len(duplicates) > 0:
                print("skip album")
                continue
            if album[0]["album id"] == 284:
                # for some reason the album with ID 284 does not exist in the full dataset
                print("skip album")
                continue
            if not self.allow_albums_with_missing_tracks:
                num_album_tracks = album[0]["album tracks"]
                if len(track_numbers) != num_album_tracks:
                    print("skip album")
                    continue
                if track_numbers[-1] != num_album_tracks:
                    print("skip album")
                    continue
            filtered_dataset_list.append(album)

        audio_features = []
        echonest_features = []
        durations = []
        track_numbers = []
        split = []
        album_indices = []
        album_lengths = []
        album_ids = []

        for album in filtered_dataset_list:
            album_indices.append(len(durations))
            album_lengths.append(len(album))
            for track in album:
                audio_features.append(torch.zeros(0))
                echonest_features.append(
                    torch.Tensor(
                        [
                            track["acousticness"],
                            track["danceability"],
                            track["energy"],
                            track["instrumentalness"],
                            track["liveness"],
                            track["speechiness"],
                            track["tempo"],
                            track["valence"],
                        ]
                    )
                )
                durations.append(track["track duration"])
                track_numbers.append(track["track number"])
                split.append(track["set split"])
            album_ids.append(track["album id"])

        audio_features = torch.stack(audio_features, dim=0)
        echonest_features = torch.stack(echonest_features, dim=0)
        durations = torch.LongTensor(durations)
        track_numbers = torch.LongTensor(track_numbers)
        split_dict = {"training": 0, "validation": 1, "test": 2}
        split = torch.Tensor([split_dict[s] for s in split]).long()
        album_indices = torch.LongTensor(album_indices)
        album_lengths = torch.LongTensor(album_lengths)
        album_ids = torch.LongTensor(album_ids)

        dataset_dict = {
            "audio_features": audio_features,
            "echonest_features": echonest_features,
            "durations": durations,
            "split": split,
            "track_numbers": track_numbers,
            "album_indices": album_indices,
            "album_lengths": album_lengths,
            "album_ids": album_ids,
        }

        torch.save(dataset_dict, self.dataset_file)
        return torch.load(self.dataset_file)


def collate_album_features_to_packed_seqs(album_feature_dict):
    keys = album_feature_dict[0].keys()
    feature_dict = {
        k: [album_feature_dict[j][k] for j in range(len(album_feature_dict))] for k in keys
    }
    sequence_lengths = torch.LongTensor([s.shape[0] for s in feature_dict["features"]])
    packed_sequences_dict = {
        k: torch.nn.utils.rnn.pack_sequence(feature_dict[k], enforce_sorted=False)
        for k in keys
    }
    packed_sequences_dict["sequence_lengths"] = sequence_lengths
    return packed_sequences_dict