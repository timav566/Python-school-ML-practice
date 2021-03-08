import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
% matplotlib
inline


class individual:

    def __init__(self, features, score=0, val_score=0):
        self.features = features
        self.score = score
        self.val_score = val_score

    def get_score(self, X_train, Y_train, X_test, Y_test, X_val, Y_val, iterations=100):
        model = CatBoostRegressor(iterations)
        dfX = pd.DataFrame(X_train)
        dfX1 = pd.DataFrame(X_test)
        dfX2 = pd.DataFrame(X_val)
        for i in range(len(self.features)):
            if (self.features[i] == 0):
                del dfX[i]
                del dfX1[i]
                del dfX2[i]
        X_train_t = dfX.values
        X_test_t = dfX1.values
        X_val_t = dfX2.values
        model.fit(X_train_t, Y_train, verbose=False)
        self.score = model.score(X_test_t, Y_test)
        self.val_score = model.score(X_val_t, Y_val)

    def mutation(self, probability):
        for i in range(len(self.features)):
            if (random.randint(0, 100) < probability):
                self.features[i] = abs(self.features[i] - 1)

    def create_new_feature(self):
        if (random.randint(0, 1) == 0):
            self.features.append(0)
        else:
            self.features.append(1)

    def delete_feature(self, i):
        self.features.pop(i)


class Dataset:

    def __init__(self, X_train, Y_train, X_test, Y_test, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val

    def update_new_feature(self, fea_a, fea_b):
        dfX1 = pd.DataFrame(self.X_train)
        dfX1['123'] = (dfX1[fea_a] + dfX1[fea_b])
        self.X_train = dfX1.values
        dfX2 = pd.DataFrame(self.X_test)
        dfX2['123'] = (dfX2[fea_a] + dfX2[fea_b])
        self.X_test = dfX2.values
        dfX3 = pd.DataFrame(self.X_val)
        dfX3['123'] = (dfX3[fea_a] + dfX3[fea_b])
        self.X_val = dfX3.values

    def delete_feature(self, num):
        dfX = pd.DataFrame(self.X_train)
        dfX.drop(dfX.loc[:, [num]], axis=1, inplace=True)
        self.X_train = dfX.values
        dfX1 = pd.DataFrame(self.X_test)
        dfX1.drop(dfX1.loc[:, [num]], axis=1, inplace=True)
        self.X_test = dfX1.values
        dfX2 = pd.DataFrame(self.X_val)
        dfX2.drop(dfX2.loc[:, [num]], axis=1, inplace=True)
        self.X_val = dfX2.values


def check_feature(current_set, num):
    flag = True
    for i in current_set:
        if (i.features[num] == 1):
            flag = False
            break
    return flag


from google.colab import files

uploaded = files.upload()

X_train = (pd.read_csv('h_10_X_train.csv')).values
X_test = (pd.read_csv('h_10_X_test.csv')).values
X_val = (pd.read_csv('h_10_X_val.csv')).values
Y_train = (pd.read_csv('h_10_Y_train.csv')['0']).values
Y_test = (pd.read_csv('h_10_Y_test.csv')['0']).values
Y_val = (pd.read_csv('h_10_Y_val.csv')['0']).values
model = CatBoostRegressor(iterations=100)
model.fit(X_train, Y_train, verbose=False)

print(model.score(X_val, Y_val))

num_of_individs = 12
best_indiv = [[0 for i in range(len(X_train[0]))] for j in range(num_of_individs)]
for i in range(num_of_individs):
    for j in range(len(X_train[0])):
        best_indiv[i][j] = 1

ds = Dataset(X_train, Y_train, X_test, Y_test, X_val, Y_val)
current_set = []
for i in range(num_of_individs):
    current_set.append(individual(best_indiv[i]))
    current_set[i].get_score(ds.X_train, ds.Y_train, ds.X_test, ds.Y_test, ds.X_val, ds.Y_val)
current_set = sorted(current_set, key=lambda student: student.score)

for i in current_set:
    print(i.val_score)

ans_features = []
ans_score = []
ans_val_score = []
best_ds = Dataset(X_train, Y_train, X_test, Y_test, X_val, Y_val)
best_features = []
best_score = 0
num_processes = 50
probability = 3
num_new_fea = 2
d = 0
for s in range(num_processes):
  if (s % 5 == 0):
    d = 0
    for i in range(len(ds.X_train[0])):
      if (check_feature(current_set, i - d) == True):
        for j in range(len(current_set)):
          current_set[j].delete_feature(i - d)
        ds.delete_feature(i - d)
        d += 1
  test_generation = []
  for i in range(len(current_set)):
    test_individual = individual(current_set[i].features.copy())
    test_generation.append(test_individual)
  for i in range(num_of_individs):
    for j in range(i + 1, num_of_individs):
      new_indiv_features = []
      for k in range(len(current_set[0].features)):
        if (random.randint(0, 1) == 0):
          new_indiv_features.append(current_set[i].features[k])
        else:
          new_indiv_features.append(current_set[j].features[k])
      test_individual = individual(new_indiv_features)
      test_generation.append(test_individual)
  for i in test_generation:
    i.mutation(probability)
  for i in range(len(current_set)):
    test_individual = individual(current_set[i].features)
    test_generation.append(test_individual)
  for i in range(num_new_fea):
    a = random.randint(0, len(current_set[0].features) - 1)
    b = random.randint(0, len(current_set[0].features) - 1)
    ds.update_new_feature(a, b)
    for g in test_generation:
      g.create_new_feature()
  for i in test_generation:
    i.get_score(ds.X_train, ds.Y_train, ds.X_test, ds.Y_test, ds.X_val, ds.Y_val)
  test_generation = sorted(test_generation, key=lambda student: student.score)
  for i in range(num_of_individs):
    current_set[i] = test_generation[len(test_generation) - i - 1]
  ans_features.append(current_set[0].features.count(1))
  ans_score.append(current_set[0].score)
  ans_val_score.append(current_set[0].val_score)
  print(current_set[0].features.count(1))
  print(len(current_set[0].features))
  print(current_set[0].score)
  print(current_set[0].val_score)
  if (current_set[0].val_score > best_score):
    best_score = current_set[0].val_score
    best_features = current_set[0].features
    best_ds = ds

ans_processes = []
for i in range(1, len(ans_score) + 1):
    ans_processes.append(i)
plt.plot(ans_processes, ans_score, 'r')
plt.xlabel('Number of processes')
plt.ylabel('Accuracy')
plt.title('Validation data')
plt.show()

ans_processes = []
for i in range(1, len(ans_score) + 1):
    ans_processes.append(i)
plt.plot(ans_processes, ans_val_score, 'r')
plt.xlabel('Number of processes')
plt.ylabel('Accuracy')
plt.title('Test data')
plt.show()
