import csv
from queue import PriorityQueue


class CityNotFoundError(Exception):
    def __init__(self, city):
        print("%s does not exist" % city)


city_1 = []
city_2 = []
kilometer = []
with open("cities.csv", "r", encoding="utf-8") as csv_file:  # to open CSV file
    readCSV = csv.reader(csv_file, delimiter=',')
    for i, j, k in readCSV:
        city_1.append(i)  
        city_2.append(j)
        kilometer.append(k)
csv_file.close()
city_1 = city_1[1:]
city_2 = city_2[1:]
kilometer = kilometer[1:]
list1 = city_1 + city_2
list1 = list(set(list1))
temp = []
first_direction = []
second_direction = []

graph_1 = {}


def build_graph():
    for x in range(len(city_1)):
        temp.append(city_1[x])
        temp.append(city_2[x])
        temp.append(kilometer[x])
        a = (temp[0], temp[1], temp[2])
        b = (temp[1], temp[0], temp[2])
        first_direction.append(a)
        second_direction.append(b)
        temp.clear()

    for start_point in list1:
        list_2 = []
        for i_1 in first_direction:
            if start_point == i_1[0]:

                list_2.append((int(i_1[2]), i_1[1]))
            elif start_point == i_1[1]:

                list_2.append((int(i_1[2]), i_1[0]))
            else:
                pass
        list_2.sort()
        a = []
        for j_1 in range(len(list_2)):
            a.append(list_2[j_1][1])

            graph_1[start_point] = a
    return graph_1


def uniform_cost_search(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    shortest = None
    for node in graph[start]:
        if node not in path:
            new_p = uniform_cost_search(graph, node, end, path)
            if new_p:
                if not shortest or len(new_p) < len(shortest):
                    shortest = new_p
    return shortest


if __name__ == "__main__":
    print(uniform_cost_search(build_graph(), input("Start Point: "), "Stop Point: "))
