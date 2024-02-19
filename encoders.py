import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


class LSTMAudioFeatureEncoder(torch.nn.Module):
    def __init__(
            self,
            num_in_features,
            hidden_size,
            num_layers,
            bidirectional=True,
            num_out_features=1,
            dropout=0.1,
    ):
        super().__init__()
        self.num_in_features = num_in_features
        self.lstm = torch.nn.LSTM(
            input_size=num_in_features,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        d = 2 if bidirectional else 1
        #self.h0 = torch.nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))
        #self.c0 = torch.nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))
        self.output_transformation = torch.nn.Linear(
            hidden_size * d * num_layers, num_out_features
        )

    def forward(self, input):
        input = input["features"]
        # input: seq_length x batch_size x 525
        # transform to (seq_length x batch_size) x 75 x 7
        feature_batch_size = input.shape[0] * input.shape[1]
        x = input.view(feature_batch_size, -1, self.num_in_features)

        output, (h_n, c_n) = self.lstm(x)
        # h_n: num_layers * num_directions x batch_size x hidden_size
        encoding = h_n.transpose(0, 1).reshape(feature_batch_size, -1)
        x = self.output_transformation(encoding)
        x = torch.sigmoid(x)
        x = x.view(input.shape[0], input.shape[1], -1)
        return x


class PrecomputedAudioFeatureEncoder(torch.nn.Module):
    def __init__(self, feature):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        self.feature_dict = {
            "acousticness": 0,
            "danceability": 1,
            "energy": 2,
            "instrumentalness": 3,
            "liveness": 4,
            "speechiness": 5,
            "tempo": 6,
            "valence": 7,
        }
        self.feature = feature
    def forward(self, x):
        if self.feature == "duration":
            return x["duration"][..., None]
        elif self.feature == "all":
            features = torch.cat([x["echonest_features"], x["duration"].unsqueeze(-1)], dim=-1)
            return features
        elif self.feature == "precomputed_learned":
            return x["learned_features"]
        else:
            features = x["echonest_features"][..., self.feature_dict[self.feature]][..., None]
            return features


class PrecomputedAudioFeaturePCAEncoder(torch.nn.Module):
    def __init__(self, num_output_features, dataset):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))
        data = []
        for example in dataset:
            data.append(torch.cat([example["echonest_features"], example["duration"].unsqueeze(-1)], dim=-1))
        data = torch.cat(data, dim=0)
        self.mean = torch.nn.Parameter(data.mean(dim=0), requires_grad=False)
        self.std = torch.nn.Parameter(data.std(dim=0), requires_grad=False)
        self.std.data[self.std.data == 0.0] = 1.0
        data = (data - self.mean) / self.std
        data = data.numpy()
        self.pca = PCA(n_components=num_output_features)
        self.pca.fit(data)

        # get transformation matrix
        self.transformation_matrix = torch.nn.Parameter(
            torch.tensor(self.pca.components_.T, dtype=torch.float32)
        )
        self.transformation_matrix.requires_grad = False

    def forward(self, x):
        features = torch.cat([x["echonest_features"], x["duration"].unsqueeze(-1)], dim=-1)
        features = (features - self.mean) / self.std
        features = torch.matmul(features, self.transformation_matrix)
        return features


class Mean(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x.mean(dim=-1)[..., None]


class OrderingLSTMEncoder(torch.nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.0
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            bidirectional=True if bidirectional else False,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        d = 2 if bidirectional else 1
        self.input_projection = torch.nn.Linear(
            in_features=input_size, out_features=hidden_size
        )
        self.output_projection = torch.nn.Linear(
            in_features=d * hidden_size * num_layers, out_features=1
        )
        self.start_embedding = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.end_embedding = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
        self.h0 = torch.nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))
        self.c0 = torch.nn.Parameter(torch.randn(d * num_layers, 1, hidden_size))

    def forward(self, padded_seqs, seq_lengths):
        # x: padded_seq_length x batch_size x num_features
        # seq_lengths: batch_size
        batch_size = padded_seqs.shape[1]
        padded_seqs = self.input_projection(padded_seqs)
        padded_seqs = F.pad(padded_seqs, [0, 0, 0, 0, 0, 1])

        # add start of sequence
        padded_seqs = torch.cat(
            [self.start_embedding.repeat(1, batch_size, 1), padded_seqs], dim=0
        )
        # add end_of sequence
        padded_seqs[
            seq_lengths, torch.arange(batch_size).to(padded_seqs.device)
        ] = self.end_embedding.repeat(1, batch_size, 1)

        narrative_features_packed_list = torch.nn.utils.rnn.pack_padded_sequence(
            padded_seqs,
            lengths=seq_lengths.cpu() + 2,
            enforce_sorted=False,
            batch_first=False,
        )

        output, (h_n, c_n) = self.lstm(
            narrative_features_packed_list,
            (self.h0.repeat(1, batch_size, 1), self.c0.repeat(1, batch_size, 1)),
        )
        encoding = h_n.transpose(0, 1).reshape(batch_size, -1)
        score = self.output_projection(encoding)
        return score, encoding