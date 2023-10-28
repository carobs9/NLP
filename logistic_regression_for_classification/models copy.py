import numpy as np
from sklearn.decomposition import PCA 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# this info was mostly retrieved from lab 2 of NLP KU course
class LrWithBPE:
    """
        1. Tokenizing the text
        2. Looking up the embedding for each token
        3. Pooling the representations to obtain a single vector for the whole document
    """
    def __init__(self, bpemb_model):
        self.bpemb_model = bpemb_model

    def get_bpemb_features(self, dataset):
        X_question = [self.bpemb_model.embed(x).mean(0) for x in dataset.values[:, 0]]  # Assuming 1st column has questions
        X_doc_text = [self.bpemb_model.embed(x).mean(0) for x in dataset.values[:, 2]]  # Assuming 3rd column has document text
        X = np.column_stack([X_question, X_doc_text])
        y = list(dataset.values[:, 3])  # Assuming 4th column is 'y'
        return X, y

    def train(self, pca_variance, X_train, y_train, X_test, y_test):
        pca = PCA(pca_variance)
        pca.fit(X_train)
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        C = np.logspace(-6, 2, 50)
        warm_start = [False, True]
        class_weight = ['balanced', None]
        
        hp = {"C": C, "warm_start": warm_start, 'class_weight': class_weight}
    
        classifier = LogisticRegression(penalty='l2', max_iter=1000)
        classifier_random = RandomizedSearchCV(
          estimator=classifier,
          param_distributions=hp,
          n_iter=100,
          cv=5,
          verbose=2,
          random_state=1000,
          n_jobs=-1,
          scoring='f1'
          )
        
        classifier.fit(X_train_pca, y_train)
        y_pred = classifier.predict(X_test_pca)

        return y_pred

    def calculate_accuracy(self, y_test, y_pred):
      accuracy = accuracy_score(y_test, y_pred)
      return accuracy

    def create_dataframe_with_preds(self, dataset, y_pred):
        dataset = pd.DataFrame(dataset, columns=["question", "answer", "document_text", "y"])
        dataset['y_pred'] = y_pred
        return dataset

    def create_report(self, y_test, y_pred):
      report = classification_report(y_test, y_pred)
      return report




class LrwithTfidf():
  """
        1. Tokenizing the text
        2. Looking up the embedding for each token
        3. Pooling the representations to obtain a single vector for the whole document
    """
  def __init__(self):
    pass

  def get_tfidf_features(self, dataset):
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
    X = vectorizer.fit_transform(dataset.values[:, 0]+ ' ' + dataset.values[:,2]) # taking the questions and doc text as features
    y = dataset.values[:, 3].astype('int')
    return X, y

  def train_classifier(self, n_components, X_train, y_train, X_test, y_test):
    svd = TruncatedSVD(n_components)
    svd.fit(X_train)
    X_train_svd = svd.transform(X_train)
    X_test_svd = svd.transform(X_test)
    lr = LogisticRegression(penalty='l2', max_iter=1000, multi_class='multinomial')
    
    C = np.logspace(-6, 2, 50)
    warm_start = [False, True]
    class_weight = ['balanced', None]
        
    hp = {"C": C, "warm_start": warm_start, 'class_weight': class_weight}
    
    classifier = LogisticRegression(penalty='l2', max_iter=1000)
    classifier_random = RandomizedSearchCV(
      estimator=classifier,
      param_distributions=hp,
      n_iter=100,
      cv=5,
      verbose=2,
      random_state=1000,
      n_jobs=-1,
      scoring='f1')
    
    lr.fit(X_train_svd, y_train) # fitting the model to the train features ('question' and 'document_text') and the train labels ('y')
    y_pred = lr.predict(X_test_svd)
    return y_pred

  def calculate_accuracy(self, y_test, y_pred):
      accuracy = accuracy_score(y_test, y_pred)
      return accuracy

  def create_dataframe_with_preds(self, dataset, y_pred):
        dataset = pd.DataFrame(dataset, columns=["question", "answer", "document_text", "y"])
        dataset['y_pred'] = y_pred
        return dataset

  def create_report(self, y_test, y_pred):
      report = classification_report(y_test, y_pred)
      return report


