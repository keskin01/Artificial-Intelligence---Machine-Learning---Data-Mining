import plotly.express as px
import random

# Do not modify the line below.
countries = ["Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Falkland Islands", "Guyana", "Paraguay",
             "Peru", "Suriname", "Uruguay", "Venezuela"]

# Do not modify the line below.
colors = ["blue", "green", "red", "yellow"]

# Write your code here
data = {
    "Argentina": ["Bolivia", "Brazil", "Chile", "Paraguay", "Uruguay"],
    "Bolivia": ["Argentina", "Brazil", "Chile", "Paraguay", "Peru"],
    "Brazil": ["Argentina", "Bolivia", "Colombia", "Guyana", "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"],
    "Chile": ["Argentina", "Bolivia", "Peru"],
    "Colombia": ["Brazil", "Ecuador", "Peru", "Venezuela"],
    "Ecuador": ["Colombia", "Peru"],
    "Guyana": ["Brazil", "Suriname", "Venezuela"],
    "Paraguay": ["Argentina", "Bolivia", "Brazil"],
    "Peru": ["Bolivia", "Brazil", "Chile", "Colombia", "Ecuador"],
    "Suriname": ["Brazil", "Guyana"],
    "Uruguay": ["Argentina", "Brazil"],
    "Venezuela": ["Brazil", "Colombia", "Guyana"],
    "Falkland Islands": []

}
list_1 = []
c = countries.copy()
paint = colors.copy()
long = 0
for i in c:
    long += len(paint)
    x = random.randrange(0, long)
    list_1.append(i)
    list_1.append(paint[x])

    for j in c:
        if j != i and j not in data.get(i):
            list_1.append(j)
            list_1.append(paint[x])
            c.remove(j)

    paint.pop(x)
    long = long * 0

dict_1 = {list_1[h]: list_1[h + 1] for h in range(0, len(list_1), 2)}
dict_1.update({"Uruguay": colors[random.randrange(len(colors))]})
dict_1.update({"Colombia": colors[random.randrange(len(colors))]})


# Do not modify this method, only call it with an appropriate argument.


# colormap should be a dictionary having countries as keys and colors as values.
def plot_choropleth(colormap):
    fig = px.choropleth(locationmode="country names",
                        locations=countries,
                        color=countries,
                        color_discrete_sequence=[colormap[c] for c in countries],
                        scope="south america")
    fig.show()


# Implement main to call necessary functions
if __name__ == "__main__":
    # coloring test
    colormap_test = dict_1

    plot_choropleth(colormap=colormap_test)
