import spacy
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC


class LemmaTokenizer(object):
    def __init__(self):
        self.en_nlp = spacy.load('en')

    def __call__(self, doc):
        doc = self.en_nlp(doc)
        return [t.lemma_ for t in doc]


class ClfTask(object):
    NameClf = {"NB": MultinomialNB, "SVC": SVC, "LinearSVC": LinearSVC,
               "LR": LogisticRegression}

    def __init__(self, clf_input, feature_input, data_path, cv=10):
        self.cv = cv
        self.clf_input = clf_input
        self.clf = self.NameClf[clf_input]()
        self.feature_input = feature_input
        self.d_text = load_files(data_path, encoding='latin1')
        self.target = self.d_text.target

    def vector2data(self, count_vect):
        X_counts = count_vect.fit_transform(self.d_text.data)
        tfidf_transformer = TfidfTransformer()
        data_tfidf = tfidf_transformer.fit_transform(X_counts)
        print(data_tfidf.shape)
        return data_tfidf

    def feature_stem(self):
        count_vect = CountVectorizer(tokenizer=LemmaTokenizer())
        return self.vector2data(count_vect)

    def feature_tfidf(self):
        count_vect = CountVectorizer()
        return self.vector2data(count_vect)

    # remove stop words
    def feature_stop_words(self):
        count_vect = CountVectorizer(stop_words='english')
        return self.vector2data(count_vect)

    def feature_stem_stop(self):
        count_vect = CountVectorizer(
            stop_words='english', tokenizer=LemmaTokenizer())
        return self.vector2data(count_vect)

    def get_feature(self):
        NameFeature = {'tfidf': self.feature_tfidf,
                       'stop_words': self.feature_stop_words,
                       'stem': self.feature_stem,
                       'stem_stop': self.feature_stem_stop}

        data = NameFeature[self.feature_input]()
        return data

    def __call__(self):
        data = self.get_feature()
        scores = cross_val_score(self.clf, data, self.target, cv=self.cv)
        print(self.clf_input)
        print(scores, '\n', scores.mean(), scores.std()*2)


if __name__ == "__main__":
    data_path = "../data/movie_review/data"
    test_clf = ClfTask('NB', 'tfidf', data_path=data_path)
    test_clf()
