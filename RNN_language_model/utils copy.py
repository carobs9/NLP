import numpy as np
import pandas as pd
# bengali tokenizer
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import WhitespaceTokenizer
from collections import defaultdict
import re

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



def get_df_lang(df, lang):
  return df[df['language'] == lang]


# Define a custom transformation function
def custom_function(value):
    # Your transformation logic here
    if value['answer_text'][0] != '':
      return value['answer_text'][0]
    return 'unanswered'


def get_tokenizer(lang):
  if lang == 'english' or lang == 'indonesian':
    return WhitespaceTokenizer() # indonesian and english tokenizers are just white space tokenizer
  elif lang == 'arabic':
    return AutoTokenizer.from_pretrained('asafaya/bert-base-arabic') # arabic tokenizer
  elif lang == 'bengali':
    return AutoTokenizer.from_pretrained('csebuetnlp/banglishbert') # bengali tokenizer

def build_vocab(data,total_vocabulary=None):
  """ Create a dictionary to store total vocabularies for different languages.
  Inputs:
      Corpuses = (dict) set of corpuses from different languages (train and val)
      Language = (str) chosen language to build de vocab on
  Outputs:
      total_vocabulary (list(str))
      train_corpus (list)
      val_corpus (list)
  """
  if not total_vocabulary:
    total_vocabulary = set()
  for text in data:
    print(text)
    text = re.sub(r'\[.*?\]', '', text) # preprocessing
    tk = get_tokenizer(lang='arabic')
    tokens = tk.tokenize(text)
    for t in tokens:
        total_vocabulary.add(t.lower())
  total_vocabulary = sorted(list(total_vocabulary))

  # Appending an empty token to 'save' the zero position for the padding token
  total_vocabulary = [""] + total_vocabulary
  return total_vocabulary

def create_embedding_matrix(tokens, embedding):
    """creates an embedding matrix from pre-trained embeddings for a new vocabulary. It also adds an extra vector
    vector of zeroes in row 0 to embed the padding token, and initializes missing tokens as vectors of 0s"""
    oov = set()
    size = embedding.emb.vector_size
    # note the extra zero vector that will used for padding
    embedding_matrix=np.zeros((len(tokens),size))
    c = 0
    for i in range(1,len(tokens)):
        try:
            embedding_matrix[i]=embedding[tokens[i]]
        except KeyError: #to catch the words missing in the embeddings
            try:
                embedding_matrix[i]=embedding[tokens[i].lower()]
            except KeyError:
                #if the token does not have an embedding, we initialize it as a vector of 0s
                embedding_matrix[i] = np.zeros(size)
                #we keep track of the out of vocabulary tokens
                oov.add(tokens[i])
                c +=1
    print(f'{c/len(tokens)*100} % of tokens are out of vocabulary')
    return embedding_matrix, oov

def text_to_indices(text, total_vocabulary, lang):
    """turns the input text (one line of text) into a vector of indices in total_vocabulary that corresponds to the tokenized words in the input text"""
    encoded_text = []
    tk = get_tokenizer(lang=lang)
    tokens = tk.tokenize(text)
    for t in tokens:
        try:
            index = total_vocabulary.index(t.lower())
            encoded_text.append(index)
        except:
            continue
    return encoded_text

def add_padding(vector, max_length, padding_index):
    """adds copies of the padding token to make the input vector the max_length size, so that all inputs are the same length (the length of the text with most words)"""
    if len(vector) < max_length:
        vector = [padding_index for _ in range(max_length-len(vector))] + vector
    return vector

def text_to_indices(text, total_vocabulary, lang):
    """turns the input text (one line of text) into a vector of indices in total_vocabulary that corresponds to the tokenized words in the input text"""
    encoded_text = []
    tk = get_tokenizer(lang=lang)
    tokens = tk.tokenize(text)
    for t in tokens:
        try:
            index = total_vocabulary.index(t.lower())
            encoded_text.append(index)
        except:
            continue
    return encoded_text

def split_sentence(size, sentence, lang):
  if lang == 'arabic': 
      sentence = sentence[::-1]

  inputs = [sentence[(i-1):i-1+size] for i in range(1,int(len(sentence))-size+1)]
  return inputs


def split_sentence_target(size, sentence, lang):
  if lang == 'arabic': 
      sentence = sentence[::-1]

  targets = [sentence[(i+size-1)] for i in range(1,int(len(sentence))-size+1)]
  return targets
