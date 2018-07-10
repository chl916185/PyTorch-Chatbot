from torchtext.data import BucketIterator, Field, TabularDataset

from pytorch_chatbot.const import DataLoaders, Datasets


def preprocess(data_path, batch_size, max_vocab_size):
    data_field, datasets = numerisize_data(data_path, max_vocab_size)
    data_loaders = create_data_generators(datasets, batch_size)
    return data_field, data_loaders


def numerisize_data(data_path, max_vocab_size):
    data_field = Field(sequential=True, use_vocab=True, eos_token='<eos>')
    train, valid, test = TabularDataset(
        data_path,
        format='tsv',
        fields=[('x', data_field), ('y', data_field)],
    ).split(split_ratio=[0.8, 0.1, 0.1])
    datasets = Datasets(train, valid, test)

    data_field.build_vocab(train, valid, test, max_size=max_vocab_size)

    return data_field, datasets


def create_data_generators(datasets, batch_size):
    train_loader, valid_loader, test_loader = BucketIterator.splits(
        (datasets.train, datasets.valid, datasets.test),
        batch_size=batch_size,
        repeat=False,
        sort_key=lambda x: len(x.sequence),
    )
    data_loaders = DataLoaders(train_loader, valid_loader, test_loader)
    return data_loaders
