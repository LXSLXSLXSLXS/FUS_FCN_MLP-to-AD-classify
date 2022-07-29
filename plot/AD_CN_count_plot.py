import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']

df = pd.read_csv('../lookupcsv3/ADNI.csv')
# print(df)
count = 0
list_ = pd.DataFrame()
for i in range(len(df)):
    label = df.loc[i, 'status']
    #print(label)
    if label == 1:
        count += 1

print("AD number:", count)

# 患病的分布情况
fig,axes = plt.subplots(1,2,figsize=(10,5))
ax = df.status.value_counts().plot(kind="bar",ax=axes[0])
ax.set_title("AD和CN受试者分布")
ax.set_xlabel("1：AD，0：CN")

df.status.value_counts().plot(kind="pie",autopct="%.2f%%",labels=['CN','AD'],ax=axes[1])
plt.show()