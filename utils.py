import os
import spacy
from nltk.sentiment.util import demo_sent_subjectivity

nlp = spacy.load('en')

neg_path = "../data/movie_review/data/neg"
pos_path = "../data/movie_review/data/pos"

sub_neg_path = "../data/movie_review/sub_data/neg"
sub_pos_path = "../data/movie_review/sub_data/pos"

punc_set = ('ADJ', 'ADV', 'CONJ', 'INTJ', 'ADP', 'DET', 'VERB', 'NOUN', 'PRON',
            'PROPN')
adv_set = ('ADJ', 'ADV', 'VERB')


def add_pos(doc):
    doc = nlp(doc)
    pos_words = []
    for word in doc:
        if word.pos_ not in adv_set:
            continue
        pos_words.append('_'.join([word.text, word.pos_]))
    pos_doc = ' '.join(pos_words)
    return pos_doc


# returen doc with part-of-speech taggings
def add_pos_dataset():
    new_neg_path = "../data/movie_review/new_pos_data/neg"
    new_pos_path = "../data/movie_review/new_pos_data/pos"

    for (old_path, new_path) in [
            (neg_path, new_neg_path), (pos_path, new_pos_path)]:
        files = os.listdir(old_path)
        for f in files:
            doc = None
            with open(os.path.join(old_path, f), encoding='latin1') as ff:
                doc = ff.read()
            pos_doc = add_pos(doc)
            file_path = os.path.join(new_path, f)
            with open(file_path, 'wt', encoding='latin1') as ff:
                ff.write(pos_doc)


# revised nltk demo_Sent_subjectivity return value
def sent_sub(sent):
    sub = demo_sent_subjectivity(sent)
    if sub == "subj":
        return True
    return False


# return doc with subjective sentences only
def add_sub():
    sub_neg_path = "../data/movie_review/sub_data/neg"
    sub_pos_path = "../data/movie_review/sub_data/pos"
    for (old_path, new_path) in [
            (neg_path, sub_neg_path), (pos_path, sub_pos_path)]:
        files = os.listdir(old_path)
        for f in files:
            with open(os.path.join(old_path, f), encoding='latin1') as ff:
                doc = ff.read()
            sub_sents = []
            sents = doc.split('\n')
            for s in sents:
                subj = sent_sub(s)
                if subj:
                    sub_sents.append(s)
            sub_doc = "\n".join(sub_sents)
            file_path = os.path.join(new_path, f)
            with open(file_path, 'wt', encoding='latin1') as ff:
                    ff.write(sub_doc)
