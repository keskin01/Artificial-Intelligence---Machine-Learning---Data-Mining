import csv
import json
import collections
import matplotlib.pyplot as plt
import re

from matplotlib import pylab


def normalize(name):
    name = name.lower()
    name = re.sub(r"[.,/#!$%^*;:{}=_`~()@]", ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


if __name__ == '__main__':
    total_data = []
    with open("part1.csv", "r", encoding="utf-8") as csv_file:  # to open CSV file
        readCSV = csv.reader(csv_file, delimiter=',')
        for row in readCSV:
            total_data.append(normalize(row[1]))  # index 1 means second row
    csv_file.close()
    with open("part2.json", "r", encoding="utf-8") as json_file:  # to open JSON file
        readJSON = json.load(json_file)
    for key, value in readJSON.items():  # key : id value : words
        total_data.append(normalize(value))
    json_file.close()
    count = collections.Counter(total_data)  # count all words from all data
    # n_bar = int(input("Top-n most frequent titles: "))
    n_bar = 10
    most_common_list = count.most_common(n_bar)  # we create
    for i in range(len(most_common_list)):
        print(most_common_list[i][0], ":", most_common_list[i][1], "times occurred.")
    xs = [x[0] for x in most_common_list]
    ys = [x[1] for x in most_common_list]
    fig = plt.figure(figsize=(16, 9))
    figure_of_window = pylab.gcf()
    figure_of_window.canvas.set_window_title('BIM476 - Data Acquisition & Processing HW1')
    plt.title('Most Frequent %s Words Graph' % n_bar, fontsize=20)
    plt.xticks(fontsize=10, rotation=0)
    plt.xlabel('Count of Words', color="red")
    plt.ylabel('Most Common Words', color="red")
    plt.barh(xs, ys, color="#00629b")
    plt.show()
