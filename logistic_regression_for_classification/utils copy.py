import numpy as np
import nltk
import string
import regex as re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('stopwords-arabic')

import pandas as pd

def get_df_lang(df, lang):
  return df[df['language'] == lang]

def build_qa_df(df):
  # 1. retrieve answers
  answers = np.array([item['answer_text'][0] for item in df['annotations'].values])
  for i in range(len(answers)):
      if answers[i] == '':
          answers[i] = 'unanswered' #  marking unanswered answers (['']) as unanswered
  # 2. retrieve questions
  questions = df['question_text']
  # 3. retrieve doc plaintext
  document_text = df['document_plaintext']
  # 4. build questions and answers dataframe
  dataframe = pd.DataFrame({'question': questions, 'answer': answers, 'document_text': document_text})
  dataframe['y'] = [0 if answer == 'unanswered' else 1 for answer in dataframe['answer']]
  return dataframe

def preprocess_text_column(df, column_name, lang):
    df[column_name] = df[column_name].str.lower()  # Convert text to lowercase
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', x)) # remove punct
    df[column_name] = df[column_name].apply(lambda x: re.sub(r'\s+', ' ', x))
    df[column_name] = df[column_name].apply(lambda x:re.sub(r' ', '', x))

    if lang == 'arabic':
        stop_words = set(nltk.corpus.stopwords.words('arabic'))
        df[column_name] = df[column_name].apply(lambda x: " ".join([word for word in str(x).split() if word not in stop_words]))

    return df

