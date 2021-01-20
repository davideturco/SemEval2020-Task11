import glob
import os
from tokenizer import NltkTokenizer, SpacyTokenizer
from tqdm import tqdm


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

            for token in tokens:
                for interval in temp_spans[0]:
                    if interval[0] <= token[1] < interval[1]:
                        label = 'I'
                        break
                    else:
                        label = 'O'
                if label != 'O':
                    if previous_label != 'O':
                        label = 'I'
                    else:
                        label = 'B'
                output.write(token[0] + '\t' + label + '\n')
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
        if not raw_spans:
            bio_encoder(encoded_folder + '/' + art_id + '.txt', tokens, spans, has_spans=False)
        else:
            bio_encoder(encoded_folder + '/' + art_id + '.txt', tokens, spans)


def remove_white_lines(directory):
    """Function to remove the empty lines from each file in a directory.
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


if __name__ == '__main__':
    create_bio_encoded_data('../datasets/train-articles/', '../datasets/train-labels-task-si', '../datasets'
                                                                                               '/train_encoded')
