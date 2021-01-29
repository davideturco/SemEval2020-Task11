from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter, SpacyTokenizer
import spacy
from data_processing import load_articles
import glob, os

TEST_FOLDER = '../datasets/test-articles'


# def get_text():
#     with open('../datasets/data_phase1/test.txt', 'r') as file:
#         text = file.read()
#         articles = text.split('\n\n')
#     return articles

def get_spans(text):
    """Given a text with spans in the form <B>, <I>, <O> returns the tokens in the span"""
    words = text.split()
    span = []
    span_sentences = []
    prev_prev_word = ''
    prev_word = ''
    for i, word in enumerate(words):
        # TEST
        if (word[0] == '.' or word[0] == '!') and len(word) != 1:
            word = word[1:]
        if word == '<B>' and prev_prev_word != '<I>':
            temp = []
            temp.append(prev_word)
        elif word == '<B>' and prev_prev_word == '<I>':
            span.append(temp)
            temp = []
            temp.append(prev_word)
        elif word == '<I>' and (prev_prev_word == '<B>' or prev_prev_word == '<I>'):
            temp.append(prev_word)
        elif word != '<I>' and (prev_prev_word == '<B>' or prev_prev_word == '<I>'):
            span.append(temp)
        elif i == (len(words) - 1) and (prev_word == '<I>' or prev_word == '<B>'):
            span.append(temp)
        prev_prev_word = prev_word
        prev_word = word
    return span


def find_sub_list(sl, l):
    """Find the begin-end word indices of a sublist inside a list
    Taken from https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list"""
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind + sll] == sl:
            results.append((ind, ind + sll - 1))
    return results


def tag_text(article_id, article_content, predicted_spans):
    """For a given article, write the predicted propaganda spans in submission format"""
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(article_content)
    tokens = [(str(x), x.idx) for x in doc if str(x) != '\n']
    spans_to_write = []
    # TEST - to remove duplicate spans
    no_duplicate = set(map(tuple, predicted_spans)) # TEST
    predicted_spans = list(map(list, no_duplicate)) # TEST
    for span in predicted_spans:
        word_indices = find_sub_list(span, [x[0] for x in tokens])
        print(article_id)
        for interval in word_indices: # TEST
            begin = tokens[interval[0]][1]
            end = tokens[interval[1]][1] + len(tokens[interval[1]][0])
            spans_to_write.append(f'{article_id}\t{begin}\t{end}\n')

    return spans_to_write


def write_submission_file(article_folder):
    article_contents, article_ids = load_articles(article_folder)
    with open('submission.txt', 'w') as output:
        for article, art_id in zip(article_contents, article_ids):
            prediction_file = '../SI-predictions/article' + art_id + '.txt'
            with open(prediction_file, 'r') as input_file:
                pred = input_file.read()
                intervals = get_spans(pred)
                if not intervals: continue
                spans_to_write = tag_text(art_id, article, intervals)
                for el in spans_to_write:
                    output.write(el)


def predict():
    model = SequenceTagger.load('../SI/SI_model.pt')
    files = sorted(glob.glob(os.path.join(TEST_FOLDER, '*.txt')))
    for file in files:
        filename = os.path.basename(file)
        with open(file, 'r') as inputs, open(os.path.join('../SI-predictions', filename), 'w') as output:
            article = inputs.read()
            splitter = SegtokSentenceSplitter(tokenizer=SpacyTokenizer('en_core_web_sm'))
            sentences = splitter.split(article)
            model.predict(sentences, mini_batch_size=16, verbose=True)
            for sentence in sentences:
                output.write(sentence.to_tagged_string())


if __name__ == '__main__':
    predict()
    write_submission_file('../datasets/test-articles')
