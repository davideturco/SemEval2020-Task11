from data_processing import *
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.visual.training_curves import Plotter
from flair.data import Sentence
import flair, torch

TRAIN_FOLDER = '../datasets/train-articles'
TRAIN_LABELS_FOLDER = '../datasets/train-labels-task-flc-tc'
TRAIN_LABELLED = '../datasets/train-labelled'
DEV_FOLDER = '../datasets/dev-articles'
DEV_LABELS_FOLDER = '../datasets/dev-labels-task-flc-tc'
DEV_LABELLED = '../datasets/dev-labelled'
DATA_FOLDER = '../datasets/data_phase2'


def create_encoded_dataset(articles_folder, label_folder, encoded_folder):
    if len(os.listdir(encoded_folder)) == 0:
        create_labelled_data(articles_folder, label_folder, encoded_folder)


def generate_datasets():
    create_encoded_dataset(TRAIN_FOLDER, TRAIN_LABELS_FOLDER, TRAIN_LABELLED)
    create_encoded_dataset(DEV_FOLDER, DEV_LABELS_FOLDER, DEV_LABELLED)

    remove_white_lines_lstm(TRAIN_LABELLED)
    remove_white_lines_lstm(DEV_LABELLED)

    generate_single_files(DEV_LABELLED, DATA_FOLDER)
    generate_single_files(TRAIN_LABELLED, DATA_FOLDER)


def train():
    generate_datasets()
    DATA_FOLDER = '../content/data'
    # MAX_TOKENS = 500
    columns = {0: 'text', 1: 'pos', 2: 'tag'}

    data_folder = DATA_FOLDER

    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train-labelled.txt',
                                  test_file='dev-labelled.txt',
                                  in_memory=False)
    # corpus._train = [x for x in corpus.train if len(x) < MAX_TOKENS]
    # corpus._test = [x for x in corpus.test if len(x) < MAX_TOKENS]

    tag_type = 'tag'
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)
    embeddings = TransformerWordEmbeddings('roberta-base', layers='-4', fine_tune=True)

    tagger: SequenceTagger = SequenceTagger(hidden_size=128,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            # dropout=0.3334816033039888,
                                            use_crf=True)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train('resources/taggers/task-TC',
                  learning_rate=0.2,
                  mini_batch_size=64,
                  max_epochs=100,
                  embeddings_storage_mode='gpu'),
    # write_weights=True)


if __name__ == '__main__':
    generate_datasets()
