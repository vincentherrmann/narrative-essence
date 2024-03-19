from main import train_model
from types import SimpleNamespace


if __name__ == "__main__":
    args = SimpleNamespace()
    args.save_model = True
    args.small_dataset = False
    args.normalize_features = True
    args.use_available_feature = False
    args.available_feature_to_use = "learned"

    args.num_negative_examples = 31
    args.patience_epochs = 20
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
