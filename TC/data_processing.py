import os, glob


# noinspection DuplicatedCode
def load_articles(article_folder)#, techniques_folder):
    """
    Function for loading the articles, the ids and the technique names
    (based on the baseline model coming with the task)
    """

    file_list = glob.glob(os.path.join(article_folder, '*.txt'))
    #techniques_list = glob.glob(os.path.join(techniques_folder, '*.txt'))
    articles_content, articles_id, technique_names = ([], [], [])

    for filename in file_list:
        with open(filename, "r", encoding="utf-8") as file:
            articles_content.append(file.read())
            articles_id.append(os.path.basename(filename).split(".")[0][7:])

    # for technique in techniques_list:
    #     with open(technique, "r") as file:
    #         propaganda_techniques_names = [line.rstrip() for line in file.readlines()]

    return articles_content, articles_id#, propaganda_techniques_names


def load_technique_spans(file):
    article_id, techniques, spans = [], [], []
    with open(file, 'r') as f:
        for line in f:
            art_id, technique, span_begin, span_end = [int(x) for x in line.rstrip().split('\t')]
            article_id.append(art_id)
            techniques.append(technique)
            spans.append((span_begin,span_end))
    return article_id, techniques, spans

