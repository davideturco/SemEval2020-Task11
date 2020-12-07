import glob
import os


# noinspection DuplicatedCode
def load_articles(article_folder, techniques_folder):
    """
    Function for loading the articles
    (based on the baseline model coming with the task)
    """

    file_list = glob.glob(os.path.join(article_folder, '*.txt'))
    techniques_list = glob.glob(os.path.join(techniques_folder, '*.txt'))
    articles_content, articles_id = ([], [])

    for filename in file_list:
        with open(filename, "r", encoding="utf-8") as file:
            articles_content.append(file.read())
            articles_id.append(os.path.basename(filename).split(".")[0][7:])

    with open(techniques_list, "r") as file:
        propaganda_techniques_names = [line.rstrip() for line in file.readlines()]

    return articles_content, articles_id, propaganda_techniques_names


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


