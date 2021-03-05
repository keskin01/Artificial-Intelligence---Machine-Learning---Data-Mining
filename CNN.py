import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

train_dataset = np.loadtxt("mnist_train.csv", delimiter=",")
test_dataset = np.loadtxt("mnist_test.csv", delimiter=",")
train_images = np.asfarray(train_dataset[:, 1:]) * (0.99 / 255) + 0.01
test_images = np.asfarray(test_dataset[:, 1:]) * (0.99 / 255) + 0.01
train_labels = np.asfarray(train_dataset[:, :1])
test_labels = np.asfarray(test_dataset[:, :1])
lr = np.arange(10)
train_labels_one_hot = (lr == train_labels).astype(np.float)
test_labels_one_hot = (lr == test_labels).astype(np.float)
train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99
test_labels_one_hot[test_labels_one_hot == 0] = 0.01
test_labels_one_hot[test_labels_one_hot == 1] = 0.99

for i in range(10):
    img = train_images[i].reshape((28, 28))
    plt.imshow(img, cmap="Greys")
    plt.show()


@np.vectorize
def activation_function(x):
    return 1 / (1 + np.e ** -x)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


class NeuralNetwork:

    def __init__(self, no_of_in_nodes, no_of_out_nodes, no_of_hidden_nodes, learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes, self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes, self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        output_vector1 = np.dot(self.wih, input_vector)
        output_hidden = activation_function(output_vector1)
        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)
        output_errors = target_vector - output_network
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = self.learning_rate * np.dot(tmp, output_hidden.T)
        self.who += tmp
        hidden_errors = np.dot(self.who.T, output_errors)
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)

    def run(self, input_vector):
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
        return output_vector

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for j in range(len(data_array)):
            result = self.run(data_array[j])
            res_max = result.argmax()
            target = labels[j][0]
            cm[res_max, int(target)] += 1
        return cm

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for k in range(len(data)):
            res = self.run(data[k])
            res_max = res.argmax()
            if res_max == labels[k]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


CNN = NeuralNetwork(no_of_in_nodes=28 * 28, no_of_out_nodes=10, no_of_hidden_nodes=100, learning_rate=0.1)

for i in range(len(train_images)):
    CNN.train(train_images[i], train_labels_one_hot[i])
for i in range(20):
    res = CNN.run(test_images[i])
    print(test_labels[i], np.argmax(res), np.max(res))

cm = CNN.confusion_matrix(train_images, train_labels)
print("confusion_matrix", "\n", cm)

for i in range(10):
    print("digit: ", i, "precision: ", precision(i, cm), "recall: ", recall(i, cm))
corrects, wrongs = CNN.evaluate(train_images, train_labels)
print("accuracy train: ", corrects / (corrects + wrongs))
pre = corrects / (corrects + wrongs)
corrects, wrongs = CNN.evaluate(test_images, test_labels)
print("accuracy: test", corrects / (corrects + wrongs))
