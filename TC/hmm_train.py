import dill
from nltk.tag import hmm
from nltk.metrics import *
from data_processing import *
from statistics import mean
from SI.hmm_train import return_in_nltk_format, f1, return_in_nltk_format_unique

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


def main():
    # TODO: put all these functions that are used for both SI and TC in a separate module (utils)
    create_encoded_dataset(TRAIN_FOLDER, TRAIN_LABELS_FOLDER, TRAIN_LABELLED)
    create_encoded_dataset(DEV_FOLDER, DEV_LABELS_FOLDER, DEV_LABELLED)

    remove_white_lines_hmm(TRAIN_LABELLED)
    remove_white_lines_hmm(DEV_LABELLED)

    generate_single_files(DEV_LABELLED, DATA_FOLDER)
    generate_single_files(TRAIN_LABELLED, DATA_FOLDER)
    # train_sequences = return_in_nltk_format(TRAIN_LABELLED)
    # test_sequences = return_in_nltk_format(DEV_LABELLED)
    train_sequences = return_in_nltk_format_unique('../datasets/data_phase2/train-labelled.txt')
    test_sequences = return_in_nltk_format_unique('../datasets/data_phase2/dev-labelled.txt')
    hmm_tagger = hmm.HiddenMarkovModelTagger.train(train_sequences, test_sequence=test_sequences)

    with open('hmm_tagger_phase2.dill', 'wb') as saved_model:
        dill.dump(hmm_tagger, saved_model)

    f1_score = []
    for sequence in test_sequences:
        sequence_words = [x[0] for x in sequence]
        tags = hmm_tagger.tag(sequence_words)
        ref = [x[1] for x in sequence]
        test = [x[1] for x in tags]
        print(ConfusionMatrix(ref,test))
        a = ConfusionMatrix(ref, test)._confusion
        f1_score.append(f1(a))
        print(f1(a))
        print(f_measure(set(ref), set(test)))
        f1_score.append(f_measure(set(ref), set(test)))
    print(f"The f1-score is {mean(f1_score)}.")


if __name__ == '__main__':
    main()
