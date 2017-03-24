from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC


class ClfTask(object):
    NameClf = {"NB": MultinomialNB, "SVC": SVC, "LinearSVC": LinearSVC,
               "log": LogisticRegression}

    def __init__(self, clf_input, feature_input, cv=10):
        self.cv = cv
        self.clf_input = clf_input
        self.clf = self.NameClf[clf_input]()
        self.feature_input = feature_input
        self.d_text = load_files("your data file path", encoding='latin1')
        self.target = self.d_text.target

    def feature_stem(self):
        pass

    def feature_tfidf(self):
        count_vect = CountVectorizer()
        X_counts = count_vect.fit_transform(self.d_text.data)
        tfidf_transformer = TfidfTransformer()
        data_tfidf = tfidf_transformer.fit_transform(X_counts)
        return data_tfidf

    def get_feature(self):
        NameFeature = {'tfidf': self.feature_tfidf}
        data = NameFeature[self.feature_input]()
        return data

    def call_clf(self):
        data = self.get_feature()
        scores = cross_val_score(self.clf, data, self.target, cv=self.cv)
        print(self.clf_input)
        print(scores, '\n', scores.mean(), scores.std()*2)


if __name__ == "__main__":
    test_clf = ClfTask('log', 'tfidf')
    test_clf.call_clf()
