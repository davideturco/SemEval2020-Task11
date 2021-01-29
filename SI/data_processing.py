import glob
import os
from tokenizer import SpacyTokenizer
from tqdm import tqdm
import nltk


# noinspection DuplicatedCode
def load_articles(article_folder):  # , techniques_folder):
    """
    Function for loading the articles
    (based on the baseline model coming with the task)
    """

    file_list = glob.glob(os.path.join(article_folder, '*.txt'))
    # techniques_list = glob.glob(os.path.join(techniques_folder, '*.txt'))
    articles_content, articles_id = ([], [])

    for filename in file_list:
        with open(filename, "r", encoding="utf-8") as file:
            articles_content.append(file.read())
            articles_id.append(os.path.basename(filename).split(".")[0][7:])

    # with open(techniques_list, "r") as file:
    #     propaganda_techniques_names = [line.rstrip() for line in file.readlines()]

    return articles_content, articles_id  # , propaganda_techniques_names


def load_spans(file):
    """
    Loads the predicted spans
    """
    article_id, span_interval = ([], [])
    with open(file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            art_id, span_begin, span_end = [int(x) for x in line.rstrip().split('\t')]
            span_interval.append((span_begin, span_end))
            article_id.append(art_id)

    return article_id, span_interval


def merge_spans(span_intervals):
    """
    If an article has overlapping spans, they are merged
    """
    temp = sorted(span_intervals)
    merge = [temp[0]]
    for el in temp:
        previous = list(merge[-1])
        if previous[1] >= el[0]:
            previous[1] = max(previous[1], el[1])
            merge.pop()
            merge.append(previous)
        else:
            merge.append(el)
    return merge


def group_spans(article_id, span_intervals):
    """
    Groups together spans from the same article
    """
    dictionary = dict.fromkeys(article_id, span_intervals)
    spans = {}
    for key, value in dictionary.items():
        spans.setdefault(key, [])
        spans[key].append(value)
    return spans


def bio_encoder(output_file, tokens, data, has_spans=True):
    """Given an article which has been tokenised and its associated propaganda spans returns a BIO-encoded file"""
    if has_spans:
        previous_label = 'O'
        global temp_spans
        with open(output_file, 'w') as output:
            for key, value in data.items():
                article_id, temp_spans = int(key), sorted(value)

            # TEST
            # for token, pos_tag in zip(tokens, pos_tags):
            for token in tokens:  # , pos_tag in zip(tokens, pos_tags):
                for interval in temp_spans[0]:
                    if interval[0] <= token[1] < interval[1] and token[0] == '.':
                        label = 'I'  # + '\n'
                        break
                    elif interval[0] <= token[1] < interval[1] and token[0] != '.':
                        label = 'I'
                        break
                    else:
                        label = 'O'
                if label != 'O':
                    if previous_label != 'O':
                        label = 'I'
                    else:
                        label = 'B'
                # TEST
                output.write(token[0] + '\t' + label + '\n')  # + pos_tag + '\n')
                previous_label = label
    else:
        with open(output_file, 'w') as output:
            for token in tokens:
                output.write(token[0] + '\t' + 'O' + '\n')


def create_bio_encoded_data(article_folder, spans_folder, encoded_folder):
    article_contents, article_ids = load_articles(article_folder)
    for article, art_id in tqdm(zip(article_contents, article_ids)):
        ids, raw_spans = load_spans(spans_folder + '/article' + art_id + '.task-si.labels')
        spans = group_spans(ids, raw_spans)
        tokenizer = SpacyTokenizer()
        tokens = tokenizer.tokenize(article)
        # TEST
        # pos_tags = nltk.pos_tag(tokens)
        # assert len(tokens) == len(pos_tags)
        if not raw_spans:
            bio_encoder(encoded_folder + '/' + art_id + '.txt', tokens, spans, has_spans=False)
        else:
            bio_encoder(encoded_folder + '/' + art_id + '.txt', tokens, spans)


def remove_white_lines_lstm(directory):
    """Function to remove the empty lines from each file in a directory (used for training LSTM).
    Taken from https://stackoverflow.com/questions/37682955/how-to-delete-empty-lines-from-a-txt-file"""
    for file in glob.glob(os.path.join(directory, '*.txt')):
        with open(file, 'r+') as infile, open(file.split('_encoded/')[0] + '_'
                                                                           'encoded/' +
                                              os.path.basename(file).split('.')[0] + '-processed.txt', 'w') as outfile:
            for line in infile:
                a = [x.strip() for x in line.split('\t')]
                if len(a) == 2 and a[0] == '': continue
                outfile.write(line)


def remove_white_lines_hmm(directory):
    """Function to remove the empty lines from each file in a directory (used for training HMM).
    Taken from https://stackoverflow.com/questions/37682955/how-to-delete-empty-lines-from-a-txt-file"""
    for file in glob.glob(os.path.join(directory, '*.txt')):
        with open(file, 'r+') as infile, open(file.split('_encoded/')[0] + '_'
                                                                           'encoded/' +
                                              os.path.basename(file).split('.')[0] + '-processed.txt', 'w') as outfile:
            for line in infile:
                if not line.strip(): continue
                a = [x.strip() for x in line.split('\t')]
                if a[0] == "": continue
                outfile.write(line)


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


def generate_test_file(origin_directory, destination_directory):
    direc = os.path.basename(origin_directory).split('-')[0]
    output_file = os.path.join(destination_directory, direc) + '.txt'
    files = sorted([x for x in glob.glob(os.path.join(origin_directory, '*.txt'))])
    with open(output_file, 'w') as output:
        for file in files:
            with open(file, 'r') as input:
                article = input.read()
                tokenizer = SpacyTokenizer()
                raw_tokens = tokenizer.tokenize(article)
                tokens = [x[0] for x in raw_tokens]
                for token in tokens:
                    if token == '\n\n' or token == '\n': continue
                    output.write(f"{token}\n")
            output.write('\n')


if __name__ == '__main__':
    create_bio_encoded_data('../datasets/train-articles/', '../datasets/train-labels-task-si', '../datasets'
                                                                                               '/train_encoded')
