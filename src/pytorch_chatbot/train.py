import argparse

from pytorch_chatbot.data import preprocess


def train(data_path, batch_size, max_vocab_size):
    data_field, data_loaders = preprocess(data_path, batch_size, max_vocab_size)

    for batch in data_loaders.train:
        x, y = batch.x, batch.y
        assert all(x, y)


def entrypoint():
    parser = argparse.ArgumentParser('train a chatbot')
    parser.add_argument('data_path')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_vocab_size', default=5000, type=int)
    args = parser.parse_args()

    train(args.data_path, args.batch_size, args.max_vocab_size)
