from data_processing import *
import os, glob
from nltk.tag import hmm

import dill

TRAIN_FOLDER = '../datasets/train-articles'
TRAIN_LABELS_FOLDER = '../datasets/train-labels-task-si'
TRAIN_ENCODED = '../datasets/train_encoded'
TEST_FOLDER = '../datasets/dev-articles'
TEST_LABELS_FOLDER = '../datasets/dev-labels-task-si'
TEST_ENCODED = '../datasets/dev_encoded'


def return_in_nltk_format(path):
    sequences = []
    for file in glob.glob(path + '/*.txt'):
        lines = []
        with open(file, 'r') as training_file:
            for line in training_file:
                a = tuple([x.strip() for x in line.split('\t')])
                lines.append(a)
        lines = list(filter(lambda x: len(x) == 2, lines))
        sequences.append(lines)
    return sequences


def return_words_in_nltk_format(path):
    sequences = []
    for file in glob.glob(path + '/*.txt'):
        lines = []
        with open(file, 'r') as training_file:
            for line in training_file:
                a = tuple([x.strip() for x in line.split('\t')])
                lines.append(a[0])
        sequences.append(lines)
    return sequences


def create_encoded_dataset(articles_folder, label_folder, encoded_folder):
    if len(os.listdir(encoded_folder)) == 0:
        create_bio_encoded_data(articles_folder, label_folder, encoded_folder)


def main():
    create_encoded_dataset(TRAIN_FOLDER, TRAIN_LABELS_FOLDER, TRAIN_ENCODED)
    create_encoded_dataset(TEST_FOLDER, TEST_LABELS_FOLDER, TEST_ENCODED)
    # TEST
    test_massimo = return_words_in_nltk_format(TRAIN_ENCODED)
    train_sequences = return_in_nltk_format(TRAIN_ENCODED)
    test_sequences = return_in_nltk_format(TEST_ENCODED)
    hmm_tagger = hmm.HiddenMarkovModelTagger.train(train_sequences, test_sequence=test_sequences)

    with open('hmm_tagger.dill', 'wb') as saved_model:
        dill.dump(hmm_tagger, saved_model)

    # TEST
    for sequence in test_massimo:
        tags = hmm_tagger.tag(sequence[1])


if __name__ == '__main__':
    main()
