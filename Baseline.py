import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import warnings
from sklearn.utils import resample
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

warnings.filterwarnings('ignore')


df = pd.read_csv('input_data/training.1600000.processed.noemoticon.csv',delimiter=',', encoding='ISO-8859-1')
df.columns = ['Sentiment','id','date','query','user','text']
df = df[['Sentiment','text']]

df['Sentiment'] = df['Sentiment'].replace({4:1})

data_cleaned = pd.DataFrame()
data_eda = pd.DataFrame()
tfidf = TfidfVectorizer()
model = LogisticRegression()

def corpus_build (df):
    """
    :Input: dataframe withe the text that needs to go throught preprocessing
        Steps:
        Converting the text into lowercase, removing unwanted tags,removing punctuations,
        lemmatising the text.
    :return: joined Word corpus and word corpus
    """

    final_corpus = []
    final_corpus_joined = []
    for i in df.index:
        text = re.sub('[^a-zA-Z]', ' ', df['text'][i])
        # Convert to lowercase
        text = text.lower()
        # remove tags
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

        # remove special characters and digits
        text = re.sub("(\\d|\\W)+", " ", text)

        ##Convert to list from string
        text = text.split()

        # Lemmatisation
        lem = WordNetLemmatizer()
        text = [lem.lemmatize(word) for word in text
                if not word in stuff_to_be_removed]
        text1 = " ".join(text)
        final_corpus.append(text)
        final_corpus_joined.append(text1)

    return final_corpus_joined, final_corpus

def get_tweets_for_model(cleaned_tokens_list):
    """
    :param cleaned_tokens_list: "input comments"
    :return: tokenized comments ex['hi', 'thus']
    """
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)

def metrics(y_train,y_train_pred,y_test,y_test_pred):
  print("training accuracy = ",round(accuracy_score(y_train,y_train_pred),2)*100)
  ConfusionMatrixDisplay.from_predictions(y_train,y_train_pred,normalize = 'all')
  print(classification_report(y_train,y_train_pred))


  print("testing accuracy = ",round(accuracy_score(y_test,y_test_pred),2)*100)
  ConfusionMatrixDisplay.from_predictions(y_test,y_test_pred,normalize = 'all')
  print(classification_report(y_test,y_test_pred))


if __name__ == '__main__':

    df_majority = df[df['Sentiment'] == 0]
    ## minority class 1
    df_minority = df[df['Sentiment'] == 1]

    df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=1234)

    df = df_majority_downsampled.append(df_minority)

    stuff_to_be_removed = list(stopwords.words('english')) + list(punctuation)
    stemmer = LancasterStemmer()

    corpus = df['text'].tolist()
    print(len(corpus))
    print(corpus[0])

    data_cleaned["text"], data_eda['text'] = corpus_build(df)
    data_cleaned["Sentiment"] = df["Sentiment"].values

    data_cleaned['Sentiment'].value_counts()

    data_eda['Sentiment'] = df["Sentiment"].values
    data_eda.head()

    # Storing positive data seperately
    positive = data_eda[data_eda['Sentiment'] == 1]
    positive_list = positive['text'].tolist()

    # Storing negative data seperately

    negative = data_eda[data_eda['Sentiment'] == 0]
    negative_list = negative['text'].tolist()

    positive_all = " ".join([word for sent in positive_list for word in sent ])
    negative_all = " ".join([word for sent in negative_list for word in sent ])

    positive_tokens_for_model = get_tweets_for_model(positive_list)
    negative_tokens_for_model = get_tweets_for_model(negative_list)

    positive_dataset = [(review_dict, "Positive") for review_dict in positive_tokens_for_model]
    negative_dataset = [(review_dict, "Negative") for review_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:333091]
    test_data = dataset[333091:]

    vector = tfidf.fit_transform(data_cleaned['text'])

    y = data_cleaned['Sentiment']
    X_train, X_test, y_train, y_test = train_test_split(vector, y, test_size=0.33, random_state=42, stratify = y)

    model.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    metrics(y_train,y_train_pred,y_test,y_test_pred)


