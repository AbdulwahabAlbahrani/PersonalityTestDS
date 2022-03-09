#%%
# general helper libraries
import os 
import numpy
import pandas as pd
import re
from math import sqrt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Flatten, SpatialDropout1D, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tqdm.notebook import tqdm
from gensim.utils import tokenize as gensim_tokenize
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks

import contractions
from collections import defaultdict

#%%
import tensorflow as tf
import tensorflow_hub as hub
print("Tensorflow version " + tf.__version__)
# from kaggle_datasets import KaggleDatasets

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu=None, zone=None, project=None, job_name='worker',
    coordinator_name=None, coordinator_address=None,
    credentials='default', service=None, discovery_url=None
).connect() # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
    print("TPU detected")
except ValueError:
    strategy = tf.distribute.MirroredStrategy() # for GPU or multi-GPU machines
    print("GPU detected")

print("Number of accelerators: ", strategy.num_replicas_in_sync)
# GCS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_PATH"
# print(GCS_PATH)

#%%
# importing the data without links in the posts, at least hopefully
data_dir = os.path.join(os.path.dirname(os.getcwd()), "PersonalityTestDS","fullpullData")

files = []

for fname in os.listdir(data_dir):
    if fname[-4:] == ".csv":
        files.append(pd.read_csv(os.path.join(data_dir,fname)))

posts_dataset =pd.concat(files,axis=0,ignore_index=True)
posts_dataset.drop('Unnamed: 0',1,inplace=True)
posts_dataset.dropna(inplace=True)
#%%
posts_dataset.info()

#%%
posts_dataset["Type"].unique()

#%%
# Inspired and changed from emot library

#%%

def convert_emojis(text):
    for emoji, meaning in UNICODE_EMO.items():
        text = text.replace(emoji, meaning)
    return text

def convert_emoticons(text):
    for emoticon, meaning in EMOTICONS.items():
        text = re.sub(emoticon, meaning, text)
        text = re.sub(emoticon.lower(), meaning, text)
    return text

def process_and_tokenize(text):
    # """
    # 1. Replace URLs in text
    # 2. Replace emojis and emoticons with their meanings
    # 3. Replace contractions with expanded forms
    # 4. Tokenize text
    # """
    # text = text.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url').strip()
    text = text.replace(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"url")
    text = convert_emojis(text)
    text = convert_emoticons(text)
    text = contractions.fix(text)
    return list(gensim_tokenize(text, lower=True))

def get_mbti_vector(mbti_type):
    I, N, F, P = mbti_type
    mbti_vector = [I == 'I', N == 'N', F == 'F', P == 'P']
    return numpy.array(mbti_vector).astype(int)



one_end = ['I', 'N', 'F', 'P']  # One slice of the indicators
opposite = ['E', 'S', 'T', 'J']  # Other end of the respective indicators
#%%
# posts_dataset['tokens'] = posts_dataset["body"].apply(process_and_tokenize)
#%%
posts_dataset[one_end] = posts_dataset.Type.apply(get_mbti_vector).tolist()
# %%
posts_dataset.head(3)
# %%

## token freq:
def get_token_freq(df):
    token_freq = defaultdict(int)
    for token_list in df.tokens:
        for token in token_list:
            token_freq[token] += 1
    token_freq = pd.Series(token_freq)
    token_freq_ser = token_freq.groupby(token_freq).count()
    token_freq_ser.name = 'NumTokens'
    token_freq_ser.index.name = 'Freq'
    return token_freq, token_freq_ser

## removing less frequent tokens
token_freq, token_freq_ser = get_token_freq(posts_dataset)
min_freq = 8
vocab = set(token_freq[token_freq > min_freq].index)
posts_dataset['tokens'] = posts_dataset.tokens.apply(lambda x: list(filter(lambda y: y in vocab, x)))
# %%
## removing small sentencess

min_tokens = 5  # min sentence length 
token_len = posts_dataset.tokens.apply(len)
dfposts = posts_dataset[token_len >= min_tokens]
# %%
def plot_num_posts_by_indicator(df):
    fig = make_subplots(rows=2, cols=2)
    for i, (one, opp) in enumerate(zip(one_end, opposite)):
        counts = df.groupby(one).Type.count().sort_index()
        row = 1 if i <= 1 else 2
        col = 1 if i in [0, 2] else 2
        fig.append_trace(go.Bar(x=[opp, one], y=counts.tolist()), row=row, col=col)
    fig.update_layout(showlegend=False, title='Number of posts split by each indicator')
    fig.show()
    
def plot_num_posts_by_type(df):
    type_posts = df.groupby('Type').body.count().sort_values(ascending=False).reset_index()
    fig = px.bar(type_posts, x='Type', y='body', title='Number of posts by MBTI type')
    fig.update_layout(xaxis_title='MBTI Type', yaxis_title='Count')
    fig.show()

plot_num_posts_by_type(posts_dataset)
plot_num_posts_by_indicator(posts_dataset)
# %%
vocab = set(w for l in posts_dataset.tokens for w in l)
vocab = {w: i + 1 for i, w in enumerate(vocab)}
posts_dataset['token_idxs'] = posts_dataset.tokens.apply(lambda l: [vocab[w] for w in l])
# %%
max_tokens = 40  # max sentence length
features = pad_sequences(posts_dataset['token_idxs'], maxlen=max_tokens, value=0., padding='pre', truncating='pre')
targets = {
    trait: posts_dataset[trait].to_numpy()
    for trait in one_end
}

class_weights = {}
for trait in one_end:
    positive_class_proportion = targets[trait].mean() 
    class_weight = {
        0: positive_class_proportion,
        1: 1 - positive_class_proportion
    }
    class_weights[trait] = class_weight
    print("class weights for", trait, ":", class_weight)

# %%
## spliting the data





# %%

plot_model(model, show_shapes=True, rankdir='LR', expand_nested=True)
# %%
cbs = [callbacks.EarlyStopping(patience=15, monitor='val_accuracy')]
histories = {}
for trait in one_end:
    x_train, x_test, y_train, y_test = train_test_splits[trait]
    history = models[trait].fit(x_train, y_train, batch_size=BS, epochs=10, validation_data=(x_test, y_test), shuffle=True, 
                                class_weight=class_weights[trait], callbacks=cbs)
    histories[trait] = history



