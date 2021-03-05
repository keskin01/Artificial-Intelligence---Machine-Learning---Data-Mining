from sklearn.model_selection import KFold, train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_distances

user_id, item_id, rating, timestamp = [], [], [], []
col = [user_id, item_id, rating, timestamp]

url = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.data'
r_cols = ['user_id', 'item_id', 'rating', 'timestamp']
dataset = pd.read_csv(url, sep='\t', names=r_cols, encoding='latin-1')
for w in range(len(col)):
    for q in dataset[r_cols[w]]:
        col[w].append(str(q))


def k_cross_validation(list_1):
    k, m = divmod(len(list_1), 10)
    total_list = list(list_1[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(10))
    test = []
    train = []
    a = 0
    for i in total_list:
        a += 1
        for j in range(0, len(i), int(len(i) * 0.2)):
            test.clear()
            train.clear()
            d = i.copy()
            test += d[j:(j + int(len(i) * 0.2))]
            del d[j:(j + int(len(i) * 0.2))]

            for x in d:
                train += x
            d.clear()
        print(str(a) + ". train set:", train, "and train len", len(train))
        print(str(a) + ". test set:", test, "and test len", len(test))


def collaborative_filtering(df):
    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]
    data_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        data_matrix[line[1] - 1, line[2] - 1] = line[3]

    user_similarity = pairwise_distances(data_matrix, metric='cosine')
    item_similarity = pairwise_distances(data_matrix.T, metric='cosine')
    print(user_similarity, item_similarity)


#
#
# def list_duplicates_of(seq, item):
#     start_at = -1
#     locs = []
#     while True:
#         try:
#             loc = seq.index(item, start_at + 1)
#         except ValueError:
#             break
#         else:
#             locs.append(loc)
#             start_at = loc
#     return locs
#
#
# source = [int(item) for item in user_id]
# for i in range(1, 943 + 1):
#     print(i, ">", list_duplicates_of(source, i))


def k_nearest_neighbors():
    # k = (10, 20, 30, 40, 50, 60, 70, 80)
    # for i in k:
    #     pass
    return None


X = dataset.drop(['rating'], axis=1)
Y = dataset['rating']

# Splitting the dataset into 80% training data and 20% testing data.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

linear_model = LinearRegression().fit(X_train, Y_train)

# Predictions on Testing data
LR_Test_predict = linear_model.predict(X_test)


def mean_absulte_error():
    mae = np.mean(np.abs((Y_test - LR_Test_predict) / Y_test)) * 100
    return mae


LR_MAE = mean_absulte_error()
print("MAE: ", LR_MAE)

k_cross_validation(rating)
collaborative_filtering(dataset)
k_nearest_neighbors()
mean_absulte_error()
