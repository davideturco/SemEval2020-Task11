from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, ELMoEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import flair, torch

TRAIN_ENCODED = '../datasets/train_encoded'
DEV_ENCODED = '../datasets/dev_encoded'
TEST_FOLDER = '../datasets/test-articles'
DATA_FOLDER = '../datasets/data_phase1'


def train():
    # flair.device = torch.device('cuda')
    columns = {0: 'text', 1: 'tag'}

    data_folder = '/content/data'

    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  test_file='dev.txt',
                                  # test_file='test.txt',
                                  in_memory=False)

    # print(len(corpus.train))
    # print(len(corpus.dev))

    tag_type = 'tag'
    # print(corpus.train[0].to_tagged_string(tag_type))
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    # print(tag_dictionary)

    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),  # ,chars_per_chunk=32),
        FlairEmbeddings('news-backward')  # ,chars_per_chunk=32)
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            # dropout=0.18995401543514967,
                                            use_crf=True)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train('resources/taggers/tags_stacking',
                  learning_rate=0.2,
                  mini_batch_size=64,
                  max_epochs=50,
                  embeddings_storage_mode='gpu',
                  write_weights=True)


if __name__ == '__main__':
    train()
