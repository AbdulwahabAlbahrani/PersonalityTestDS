#%%
import pandas as pd
#%%
# This file is now unavaliable as it is too big to
# upload to github
#data_set_one = pd.read_csv("MBTI500.csv")

#%%
# data_set_one.info()
# %%

# %%
# becuse the file was big, I needed to devid it into
# smalled ones to be able to upload it to github
# C1 = data_set_one[1:35336]
# C2 = data_set_one[35336:70672]
# C3 = data_set_one[70672:1060068]
# # %%
# C1.info()
# # %%
# C2.info()

# # %%
# C3.info()
# # %%
# C1.to_csv("posts1.csv")
# C2.to_csv("posts2.csv")
# C3.to_csv("posts3.csv")
# %%
# load files
c1 = pd.read_csv("data/posts1.csv")
c2 = pd.read_csv("data/posts2.csv")
c3 = pd.read_csv("data/posts3.csv")

brithdays= pd.read_csv("data/MBIT_birthdays.csv")

# %%
frams = [c1,c2,c3]
data_frame = pd.concat(frams)
data_frame.info()
# %%
# droping the Unnamed:0 colume that was generated during saving the data
data_frame=data_frame.drop("Unnamed: 0",1)
data_frame.info()

# %%
brithdays.info()
# %%
brithdays.head(5)
# %%
