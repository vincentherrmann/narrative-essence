from main import train_model
from types import SimpleNamespace


if __name__ == "__main__":
    args = SimpleNamespace()
    args.save_model = False
    args.small_dataset = True
    args.normalize_features = True
    args.include_learned_feature = True
    args.song_features = "all"
    # options:
    # "acousticness"
    # "danceability"
    # "energy"
    # "instrumentalness"
    # "liveness"
    # "speechiness"
    # "tempo"
    # "valence"
    # "duration"
    # "precomputed_learned"
    # "learned_frozen"
    # "learned"
    # "pca"

    args.num_negative_examples = 31
    args.patience_epochs = 50
    args.feature_encoder_type = "lstm"
    args.feature_encoder_hidden_units = 128
    args.feature_encoder_num_layers = 2
    args.feature_encoder_dropout = 0.1
    args.num_encoding_features = 1

    args.ordering_encoder_hidden_units = 32
    args.ordering_encoder_num_layers = 2
    args.ordering_encoder_bidirectional = True
    args.ordering_encoder_dropout = 0.0
    args.ordering_encoder_weight_decay = 1e-5

    print("train narrative essence extractor")
    train_model(args)