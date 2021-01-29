import os, glob
from tokenizer import SpacyTokenizer
from tqdm import tqdm
import nltk

# noinspection DuplicatedCode
def load_articles(article_folder):
    """
    Function for loading the articles, the ids and the technique names
    (based on the baseline model coming with the task)
    """

    file_list = glob.glob(os.path.join(article_folder, '*.txt'))
    articles_content, articles_id, technique_names = ([], [], [])

    for filename in file_list:
        with open(filename, "r", encoding="utf-8") as file:
            articles_content.append(file.read())
            articles_id.append(os.path.basename(filename).split(".")[0][7:])

    return articles_content, articles_id


def load_technique_spans(file):
    article_id, techniques, spans = [], [], []
    with open(file, 'r') as f:
        for line in f:
            art_id, technique, span_begin, span_end = [x for x in line.rstrip().split('\t')]
            article_id.append(art_id)
            techniques.append(technique)
            spans.append((int(span_begin), int(span_end)))
    return article_id, techniques, spans


def group_techniques(article_id, techniques, spans):
    """Create dictionary for each article containing propaganda spans and associated propaganda labels"""
    dictionary = dict.fromkeys(article_id, [x for x in zip(techniques, spans)])
    spans = {}
    for key, value in dictionary.items():
        spans.setdefault(key, [])
        spans[key].append(value)
    return spans


def encode_data(output_file, tokens, data, has_spans=True):
    """Attach propaganda labels to tokens"""
    if has_spans:
        with open(output_file, 'w') as output:
            for key, value in data.items():
                article_id, temp = int(key), sorted(value[0], key=lambda x: x[1])
                technique = [x[0] for x in temp]
                temp_spans = [x[1] for x in temp]
            for token in tokens:
                for prop, interval in zip(technique, temp_spans):
                    if interval[0] <= token[1] < interval[1] and token[0] == '.':  # TEST
                        label = prop + '\n'  # TEST
                        break  # TEST
                    elif interval[0] <= token[1] < interval[1] and token[0] != '.':  # TEST
                        label = prop
                        break
                    elif token[0] == '.':
                        label = 'O\n'
                    else:
                        label = "O"
                output.write(token[0] + '\t' + label + '\n') # pos_tag[1]
    else:
        with open(output_file, 'w') as output:
            for token in tokens:
                output.write(token[0] + '\t' + 'O' + '\n') # pos_tag[1]


def create_labelled_data(article_folder, labels_folder, labelled_folder):
    article_contents, article_ids = load_articles(article_folder)
    for article, art_id in tqdm(zip(article_contents, article_ids)):
        ids, techniques, raw_spans = load_technique_spans(labels_folder + '/article' + art_id + '.task-flc-tc.labels')
        spans = group_techniques(ids, techniques, raw_spans)
        tokenizer = SpacyTokenizer()
        tokens = tokenizer.tokenize(article)
        # TEST
        pos_tags = nltk.pos_tag([x[0] for x in tokens])
        assert len(tokens) == len(pos_tags)
        if not raw_spans:
            encode_data(labelled_folder + '/' + art_id + '.txt', tokens, spans, has_spans=False)
        else:
            encode_data(labelled_folder + '/' + art_id + '.txt', tokens, spans)


def remove_white_lines_lstm(directory):
    """Function to remove the empty lines from each file in a directory."""
    for file in glob.glob(os.path.join(directory, '*.txt')):
        with open(file, 'r+') as infile, open(file.split('labelled/')[0] +
                                              'labelled/' +
                                              os.path.basename(file).split('.')[0] + '-processed.txt', 'w') as outfile:
            for line in infile:
                a = [x.strip() for x in line.split('\t')]
                if len(a) == 3 and a[0] == '': continue # 2 is POS are not used
                outfile.write(line)


def remove_white_lines_hmm(directory):
    """Function to remove the empty lines from each file in a directory.
    Taken from https://stackoverflow.com/questions/37682955/how-to-delete-empty-lines-from-a-txt-file"""
    for file in glob.glob(os.path.join(directory, '*.txt')):
        with open(file, 'r+') as infile, open(file.split('labelled/')[0] +
                                              'labelled/' +
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
                #output.write('\n') # remove for transformer


if __name__ == '__main__':
    create_labelled_data('../datasets/dev-articles', '../datasets/dev-labels-task-flc-tc', '../datasets/dev-labelled')
