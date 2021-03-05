import csv
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import pylab


# This code is only a template, feel free to modify it as you like.
# Do not use statistics library when calculating mean, median, and mode.

# Read and pre-process data

def preprocess():
    d = pd.read_csv('grades.csv', keep_default_na=False)
    d.drop_duplicates(inplace=True, keep='first')
    d.to_csv('grades1.csv', index=False)
    total_data = []

    with open("grades1.csv", "r", encoding="utf-8") as csv_file:  
        readCSV = csv.reader(csv_file, delimiter=',')

        for key in readCSV:
            total_data.append(key[1])

    csv_file.close()

    midterm = []
    for i in total_data:
        if i == 'na' or i == 'NA':
            pass
        else:
            midterm.append(float(i))

    return total_data, midterm


# Write your own code to calculate the mean of grades.
def calculate_mean():
    return sum(preprocess()[1]) / len(preprocess()[1])


# Write your own code to calculate the median of grades.
def calculate_median():
    median = preprocess()[1]
    median.sort()
    if len(median) % 2 == 0:

        return ((median[len(median) // 2]) + (median[len(median) // 2 - 1])) / 2
    else:
        return median[len(median) // 2]


# Write your own code to calculate the mode of grades.
def calculate_mode():
    mode = preprocess()[1]
    mode.sort()
    n = len(mode)
    data = Counter(mode)
    get_mode = dict(data)
    mode = [k for k, v in get_mode.items() if v == max(list(data.values()))]

    if len(mode) == n:
        "No mode found"
    else:
        get_mode = ', '.join(map(str, mode))

        return get_mode


def boxplot():
    plt.figure(figsize=(16, 9))
    figure_of_window = pylab.gcf()
    figure_of_window.canvas.set_window_title('BIM476 - Data Acquisition & Processing HW2')
    plt.title('Student-Grade Boxplot', fontsize=20)
    plt.xticks(fontsize=10, rotation=0)
    plt.xlabel('Grade', color="red")
    plt.ylabel('Students', color="red")
    plt.boxplot(preprocess()[1])
    plt.show()


# Visualization by histogram
def histogram():
    plt.figure(figsize=(16, 9))
    figure_of_window = pylab.gcf()
    figure_of_window.canvas.set_window_title('BIM476 - Data Acquisition & Processing HW2')
    plt.title('Student-Grade Histogram', fontsize=20)
    plt.xticks(fontsize=10, rotation=0)
    plt.xlabel('Grade', color="red")
    plt.ylabel('Students', color="red")
    plt.hist(preprocess()[1], bins=10, color="#00629b")
    plt.show()


# Print or tabulate the answers of Q1, Q2, Q3, Q4, and Q5.
def summary():
    return print("", len(preprocess()[0]) - 1, "students take this course\n", len(preprocess()[1]) - 1,
                 "of the students attended the first midterm\n", "Mean is:", calculate_mean(), "\n", "Median is:",
                 calculate_median(), "\n", "Mode is:", calculate_mode())


if __name__ == '__main__':
    summary()
    boxplot()
    histogram()
