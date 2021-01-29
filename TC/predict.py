from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter, SpacyTokenizer
import spacy
from data_processing import load_articles
import re
import glob, os

TEST_FOLDER = '../datasets/test-articles'
TECHNIQUES = ['<Whataboutism,Straw_Men,Red_Herring>', '<Flag-Waving', '<Loaded_Language>', '<Causal_Oversimplification>',
              '<Appeal_to_fear-prejudice>', '<Appeal_to_Authority>', '<Doubt>', '<Name_Calling,Labeling>', '<Exaggeration,Minimisation>',
              '<Thought-terminating_Cliches>', '<Slogans>', '<Bandwagon,Reductio_ad_hitlerum>', '<Black-and-White_Fallacy>',
              '<Repetition>']

def get_spans(text):
    """Given a text with technique lable attaced in the form "word <label>" return the tokens in the span"""
    words = text.split()
    span = []
    prev_prev_word = ''
    prev_word = ''
    for i, word in enumerate(words):
        # TEST
        if (word[0] == '.' or word[0] == '!') and len(word) != 1:
            word = word[1:]
        if word in TECHNIQUES and prev_prev_word not in TECHNIQUES:
            temp = []
            temp.append((prev_word, re.findall(r'<(.+?)>', word)))
        elif word in TECHNIQUES and prev_prev_word in TECHNIQUES:
            temp.append((prev_word, re.findall(r'<(.+?)>', word)))
        elif word in TECHNIQUES and prev_prev_word in TECHNIQUES and word != prev_prev_word:
            span.append(temp)
            temp = []
            temp.append((prev_word, re.findall(r'<(.+?)>', word)))
        elif word not in TECHNIQUES and (prev_prev_word in TECHNIQUES):
            span.append(temp)
        elif i == (len(words) - 1) and prev_word in TECHNIQUES:
            span.append(temp)
        prev_prev_word = prev_word
        prev_word = word
    return span


def find_sub_list(sl, l):
    # Taken from https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list
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
    # TEST
    tokens = [(str(x), x.idx) for x in doc if str(x) != '\n']
    spans_to_write = []
    # TEST - to remove duplicate spans
    # no_duplicate = set(map(tuple, predicted_spans))  # TEST
    # predicted_spans = list(map(list, no_duplicate))  # TEST
    for span in predicted_spans:
        # TODO: deal with repeated span intervals such as "witch hunt" in 833052347
        span_tokens = [y[0] for y in span]
        techniques = [y[1] for y in span]
        word_indices = find_sub_list(span_tokens, [x[0] for x in tokens])
        print(article_id)
        for interval in word_indices:  # TEST
            begin = tokens[interval[0]][1]
            end = tokens[interval[1]][1] + len(tokens[interval[1]][0])
            spans_to_write.append(f'{article_id}\t{techniques[0][0]}\t{begin}\t{end}\n')

    return spans_to_write


def write_submission_file(article_folder):
    article_contents, article_ids = load_articles(article_folder)
    with open('submission.txt', 'w') as output:
        for article, art_id in zip(article_contents, article_ids):
            prediction_file = '../TC-predictions/article' + art_id + '.txt'
            with open(prediction_file, 'r') as input_file:
                pred = input_file.read()
                intervals = get_spans(pred)
                # if not intervals: continue
                spans_to_write = tag_text(art_id, article, intervals)
                for el in spans_to_write:
                    output.write(el)


def predict():
    model = SequenceTagger.load('finale.pt')
    files = sorted(glob.glob(os.path.join(TEST_FOLDER, '*.txt')))
    for file in files:
        filename = os.path.basename(file)
        with open(file, 'r') as inputs, open(os.path.join('../TC-predictions', filename), 'w') as output:
            article = inputs.read()
            splitter = SegtokSentenceSplitter(tokenizer=SpacyTokenizer('en_core_web_sm'))
            sentences = splitter.split(article)
            model.predict(sentences, mini_batch_size=16, verbose=True)
            for sentence in sentences:
                output.write(sentence.to_tagged_string())


if __name__ == '__main__':
    predict()
    write_submission_file('../datasets/test-articles')
