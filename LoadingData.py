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
C1 = data_set_one[0:15000]
C2 = data_set_one[15001:30000]
C3 = data_set_one[30001:45000]
C4 = data_set_one[45001:60000]
C5 = data_set_one[60001:75000]
C6 = data_set_one[75001:90000]
C7 = data_set_one[105001:106067]
# %%
C1.info()
#%%
C2.info()

# %%
C3.info()
#%%
C4.info()


# %%
C1.to_csv("data/posts1.csv")
C2.to_csv("data/posts2.csv")
C3.to_csv("data/posts3.csv")
C4.to_csv("data/posts4.csv")
C5.to_csv("data/posts5.csv")
C6.to_csv("data/posts6.csv")
C7.to_csv("data/posts7.csv")

# %%
# load files
c1 = []

k = 6
i = 1
while i < k:
    c1.append(pd.read_csv("data/posts"+str(i)+".cvs"))
    i = i + 1
#%%
brithdays= pd.read_csv("data/MBIT_birthdays.csv")

# %%
frams = [c1,c2,c3]
#%%
data_set_one = pd.concat(c1)
data_set_one.info()
# %%
# droping the Unnamed:0 colume that was generated during saving the data
data_frame=data_set_one.drop("Unnamed: 0",1)
data_frame.info()

# %%
brithdays.info()
# %%
brithdays.head(5)
# %%
