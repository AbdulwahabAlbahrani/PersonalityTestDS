#%%
# general helper libraries
from ftplib import all_errors
from posixpath import dirname
import numpy as np
import pandas as pd
import os
import re

# deep learning functionality
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import models
#%%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#%%
# importing the data without links in the posts, at least hopefully
data_dir = os.path.join(os.path.dirname(os.getcwd()), "PersonalityTestDS","data")

files = []

for fname in os.listdir(data_dir):
    if fname[-4:] == ".csv":
        files.append(pd.read_csv(os.path.join(data_dir,fname)))

posts_dataset =pd.concat(files,axis=0,ignore_index=True)

#%%
posts_dataset.info()
#%%
# posts_dataset.drop('Unnamed: 0',1,inplace=True)

#%%
posts_dataset.info()

#%%
posts_dataset.dropna(inplace=True)

#%%
## Most are implementrd inthe cleaning_data.py
# posts_dataset['body'] = [posts_dataset.body[i].lower () for i in range(len(posts_dataset['body']))]

# trim white spaces
# posts_dataset.body = [posts_dataset.body[item].strip() for item in range ( len( posts_dataset.body ) ) ]
#%%
# break text up into individual words    len(posts_dataset['body'])
# posts_dataset['body'] = [print(posts_dataset["body"][i]) for i in range(5)]


#%%
# a function that I created to index the input based on it's position in an input dataset

def what_type_is_it (this_lable, all_lables):
        for k in range(len(all_lables)):
            if this_lable == all_lables[k]:
                return k
                
#%%

texts = []
labls = []


#%%
all_lables_ =  [ "INTP", "INTJ", "ENTP","INFJ",
            "INFP", "ENFP", "ISTP", "ENTJ",
            "ENFJ", "ISTJ", "ESTP", "ISFP",
            "ESFP", "ISFJ", "ESTJ", "ESFJ" ]


for i in posts_dataset["Type"]:
    labls.append(what_type_is_it(i,all_lables_) )
#%%
for i in posts_dataset["body"]:
    texts.append(i)

#%%
# posts_dataset.body = [posts_dataset.body[item].strip() for item in range ( len( posts_dataset.body ) ) ]
splited_texts = []
for i in texts:
    i = i.strip()
    i = i.split()
    splited_texts.append(i)

# %%
splited_texts[649]
# %%
posts_dataset.body.isna().value_counts()
# %%
top_n_words = 200000

tokenizer = Tokenizer(num_words = top_n_words, oov_token="<00V>")
tokenizer.fit_on_texts(texts)

#%%
list(tokenizer.word_index.items())[:6]
# %%
sequences = tokenizer.texts_to_sequences(texts)
# sequences = pad_sequences(sequences)
# The vectorized first review
sequences[0]
# %%

# recreate first review
" ".join([list(tokenizer.word_index)[word_index-1] for word_index in sequences[2]])


# %%

max_len = 150
features = pad_sequences(sequences, maxlen=max_len)
# %%
features[5]


# %%
# Model training
## create random indices
indices = np.arange(features.shape[0])
np.random.shuffle(indices)

## randomize our data
x_train = features[indices]
y_train = np.asarray(labls)[indices]
# %%
model = models.Sequential()
model.add(layers.Embedding(input_dim=top_n_words, input_length=max_len, output_dim=32))
model.add(layers.Flatten())
model.add(layers.Dense(units=1, activation='sigmoid'))
model.summary()
# %%

# opt = adam  rmsprop

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)




# %%
results = pd.DataFrame(history.history)

# plot loss
results[['val_loss', 'loss']].plot(figsize=(8, 5))
# %%


# plot accuracy
results[['val_accuracy', 'accuracy']].plot(figsize=(8, 5))


# %%
