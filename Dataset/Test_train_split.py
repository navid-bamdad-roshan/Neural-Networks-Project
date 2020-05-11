import pandas as pd
from sklearn.utils import shuffle

dataset = pd.read_csv("final_dataset.csv", encoding='utf-8')

dataset = shuffle(dataset)

size = dataset.shape[0]

train_size = int((size*80)/100)

train_dataset = dataset.iloc[0:train_size]
test_dataset = dataset.iloc[train_size:]

print(dataset.shape[0])
print(train_dataset.shape[0])
print(test_dataset.shape[0])

train_dataset.to_csv("train_dataset.csv", index=False, header=True, encoding="utf-8")
test_dataset.to_csv("test_dataset.csv", index=False, header=True, encoding="utf-8")