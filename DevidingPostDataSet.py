#%%
import pandas as pd

# %%
DataSet = pd.read_csv("data/MBTI500.csv")
# %%
DataSet.info()
# %%

dataset_size=DataSet.size/2
i = 0
n = 1
k = 12
l = round(dataset_size/k)
while n <= k:
    j = l*n
    print(j)
    DataSet[0:j].to_csv("data/posts"+str(n)+".cvs")
    n = n + 1


#%%
print("data/posts"+str(n)+".cvs")
# %%
