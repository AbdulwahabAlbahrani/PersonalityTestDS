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

posts_dataset['body'] = posts_dataset['body'].str.replace('&gt;',"")
posts_dataset['body'] = posts_dataset['body'].str.replace("I\'m;","I'm")
posts_dataset['body'] = posts_dataset['body'].str.replace("it\'s;","it's")
posts_dataset['body'] = posts_dataset['body'].str.replace("they\'re","they're")

posts_dataset['body'] = posts_dataset['body'].str.replace('"',"")
posts_dataset['body'] = posts_dataset['body'].str.replace("'","")


#%%
# droping the colume that is added when we exported the dataset
posts_dataset.drop('Unnamed: 0',1,inplace=True)

#%%
# a function that I created to index the input based on it's position in an input dataset

def what_type_is_it (this_lable, all_lables):
        for k in range(len(all_lables)):
            if this_lable == all_lables[k]:
                return k
                
        

# %%

## this was to figure out how to give each type a number
# my_lables =["INTJ","INTP", "ISTJ" ]
# new_my_lables =[]



# for i in my_lables:
#     # for k in range(len(lables)):
#     #     if i == lables[k]:
#     #         new_my_lables.append(k)
#     new_my_lables.append(what_type_is_it(lables,i))

# new_my_lables

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

# %%
for i in posts_dataset["body"]:
    texts.append(i)
# %%
texts
# %%
labls
# %%

# %%

# %%
labls
# %%
