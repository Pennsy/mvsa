# Movie Review Sentiment Analysis
By: @Pennsy

This project examines different text features for sentiment classifier on movie review data.
The methods used for the classifier are Na&iuml;ve Bayes, SVM with linear kernel and Logistic Regression. The features considered are unigrams with tf*idf, lemmatization, stop words and Part-of-Speech tagging.

Most of the work in the project refers the work done in (Bo, 2002). Besides reproduce their work, this study evaluates the effectiveness of Part-of-Speech tagging on polarity classification.

The movie review dataset used in this project is
[The Polarity Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (Bo, 2004).


## Support Library
SpaCy

nltk

Scikit-learn

## Reference
Bo Pang, Lillian Lee, and Shivakumar Vaithyanathan, Thumbs up? Sentiment Classification using Machine Learning Techniques, *Proceedings of EMNLP 2002*.

Bo Pang and Lillian Lee, A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on Minimum Cuts, *Proceedings of ACL 2004*.
