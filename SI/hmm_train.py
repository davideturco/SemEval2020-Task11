import glob
import os

import dill
from nltk.tag import hmm
from nltk.metrics import *
import numpy as np
from data_processing import *
from statistics import mean

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


def create_encoded_dataset(articles_folder, label_folder, encoded_folder):
    if len(os.listdir(encoded_folder)) == 0:
        create_bio_encoded_data(articles_folder, label_folder, encoded_folder)


def f1(confusion):
    """Given a confusion matrix it returns the f1-score."""
    A = np.array(confusion)

    precision = np.empty([3, 1])
    recall = np.empty([3, 1])
    true_pos_neg = np.empty([3, 1])
    diagonal = 0.
    for i in range(3):
        den_precision = 0.
        den_recall = 0.
        true_pos_neg[i] = A[i][i]
        diagonal += true_pos_neg[i]
        for j in range(3):
            den_precision += A[j][i]
            den_recall += A[i][j]

        if den_precision == 0. or den_recall == 0.:
            return 0.
        else:
            precision[i] = true_pos_neg[i] / den_precision
            recall[i] = true_pos_neg[i] / den_recall

    weights = true_pos_neg / diagonal
    averaged_precision = np.average(precision, weights=weights)
    averaged_recall = np.average(recall, weights=weights)

    f1_score = 2 * (averaged_precision * averaged_recall) / (averaged_precision + averaged_recall)
    return f1_score


def main():
    create_encoded_dataset(TRAIN_FOLDER, TRAIN_LABELS_FOLDER, TRAIN_ENCODED)
    create_encoded_dataset(TEST_FOLDER, TEST_LABELS_FOLDER, TEST_ENCODED)

    train_sequences = return_in_nltk_format(TRAIN_ENCODED)
    test_sequences = return_in_nltk_format(TEST_ENCODED)
    hmm_tagger = hmm.HiddenMarkovModelTagger.train(train_sequences, test_sequence=test_sequences)

    with open('hmm_tagger.dill', 'wb') as saved_model:
        dill.dump(hmm_tagger, saved_model)

    # TEST
    f1_score = []
    for sequence in test_sequences:
        sequence_words = [x[0] for x in sequence]
        # TODO: maybe save tags on a txt file. Easier to visualise and compare results
        tags = hmm_tagger.tag(sequence_words)
        ref = [x[1] for x in sequence]
        test = [x[1] for x in tags]
        a = ConfusionMatrix(ref, test)._confusion
        f1_score.append(f1(a))
        print(f1(a))
    # TODO: investigate if result is actually correct. F1 score seems too high
    print(f"The f1-score is {mean(f1_score)}.")


if __name__ == '__main__':
    main()
