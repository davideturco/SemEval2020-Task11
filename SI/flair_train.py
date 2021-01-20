from data_processing import remove_white_lines
import glob, os
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

TRAIN_ENCODED = '../datasets/train_encoded'
TEST_ENCODED = '../datasets/dev_encoded'
DATA_FOLDER = '../datasets/data_phase1'


def generate_single_files(origin_directory, destination_directory):
    """Create single text file for given directory (such as train, dev, test) in line with Flair requirements"""
    direc = os.path.basename(origin_directory).split('_')[0]
    output_file = os.path.join(destination_directory, direc) + '.txt'
    files = sorted([x for x in glob.glob(os.path.join(origin_directory, '*-processed.txt'))])

    with open(output_file, 'w') as output:
        for file in files:
            with open(file, 'r') as input:
                lines = input.read()
                output.write(lines)
            output.write('\n')


def main():
    # remove_white_lines(TRAIN_ENCODED)
    # remove_white_lines(TEST_ENCODED)
    generate_single_files(TRAIN_ENCODED,DATA_FOLDER)
    generate_single_files(TEST_ENCODED,DATA_FOLDER)

    columns = {0: 'text', 1: 'tag'}

    data_folder = DATA_FOLDER

    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  dev_file='dev.txt',
                                  in_memory=False)
    # print(len(corpus.train))
    # print(len(corpus.dev))

    tag_type = 'tag'
    # print(corpus.train[0].to_tagged_string(tag_type))
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    # print(tag_dictionary)

    embedding_types = [
        WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward')
    ]

    embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

    tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)

    trainer: ModelTrainer = ModelTrainer(tagger, corpus)

    trainer.train('resources/taggers/tags_stacking',
                  learning_rate=0.1,
                  mini_batch_size=1,
                  max_epochs=150,
                  embeddings_storage_mode='gpu')


if __name__ == '__main__':
    main()
