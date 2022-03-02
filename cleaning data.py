#%%
from dataclasses import dataclass
from decimal import ROUND_DOWN
from math import floor
import pandas as pd
import plotly.express as px
import numpy as np
import math
#%%
# This file is now unavaliable as it is too big to
# upload to github
#data_set_one = pd.read_csv("MBTI500.csv")

#%%

j = []
for i in range(50):
    j.append(pd.read_csv("fullpullData/file" + str(i)+".csv"))

#%%
posts_data = pd.concat(j,axis=0,ignore_index=True)

#%%
posts_data.drop('Unnamed: 0',1,inplace=True)
#%%
posts_data.info()

#%%
type_value_counts = posts_data['Type'].value_counts()
type_value_counts
#%%

# new_type_value_counts = pd.DataFrame({'index':type_value_counts.index,'value':type_value_counts.values})
fig = px.bar(type_value_counts)
fig.show()
#%%

# df = px.data.iris()
# fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species")
# fig.show()
# %%

posts_data.head(10)
# %%
posts_data['body'][1183636]
# %%
# removing \n from every post pody
posts_data['body'] = posts_data['body'].str.replace('\n',"")

# %%
# remove the links that were made usign the the markdown way
# that put the link inside () and the text it appears is under []

posts_data['body'] = posts_data['body'].str.replace(r'\(.*\)',"")
posts_data['body'] = posts_data['body'].str.replace(r'\[.*\]',"")

#%%

# %%
## exporting the cleaned dataset


n = 0
k = 49
l = floor(posts_data.shape[0]/k)
while n <= k+1:
    lower = l*n
    higher = l*(n+1)
    print(str(lower)+":"+str(higher))
    posts_data[lower:higher].to_csv("data/posts"+str(n)+".csv")
    n = n + 1


# %%
