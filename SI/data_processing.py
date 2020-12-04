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

def load_spans(spans_folder):