import math
import random
import sys

import networkx as nx
from joblib.numpy_pickle_utils import xrange
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

import random
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

from GridSearchSARIMA import GridSearchSARIMA


def transform(time):
    return [[0, 1, time % 2],
            [0, 0, 1],
            [0, 0, 0]]


def noise(adj):
    for i in range(len(adj)):
        adj[i] += np.random.normal(0, 0.2)


def outlier(adj, choices):
    outliers = []

    for i in choices:
        pos = random.randint(0, len(adj) - 1)
        size = int(math.sqrt(len(adj)))
        adj[pos][i] += 1
        outliers.append((math.floor(pos / size), pos % size))

    return outliers


def darken_color(color, amount=0.8):  # amount is percantage.
    rgb = mcolors.hex2color(color)  # Changing the format.
    hsv = mcolors.rgb_to_hsv(rgb)  # Changing the format again.
    new_value = hsv[2] * amount  # Darkening the colour.
    new_rgb = mcolors.hsv_to_rgb((hsv[0], hsv[1], new_value))  # Changing back the format.
    new_hex = mcolors.rgb2hex(new_rgb)  # Changing back the format again.
    return new_hex


# A helper function for adding the edges to the multigraph with colors and edge angles.
def add_edge(G, a, b, edgeColor=0):
    # If the edge is already in the graph we must ad an angle to the edge, so there is no overlap.
    if (a, b) in G.edges:
        max_rad = max(x[2]['rad'] for x in G.edges(data=True) if sorted(x[:2]) == sorted([a, b]))
    else:  # If the edge isn't in the graph it can be straight.
        max_rad = 0
    G.add_edge(a, b, rad=max_rad + 0.1, color=edgeColor)  # Adding the edge with angle and colour.


def createGraph(num):
    G = nx.Graph()  # Creating an empty networkx graph.

    # Adding all of the nodes.
    G.add_nodes_from(range(num))

    return G


def createMultigraph(num, paths):
    G_multi = nx.MultiDiGraph()  # Creates an empty graph.
    G_multi.add_nodes_from(range(num))

    # Creating random colours so we can use them for showing paths.
    valid_colors = list(mcolors.CSS4_COLORS.values())  # List of valid color names

    # Iterating through pairs of colours and paths, and adding them to the graph.
    for color, path in zip(valid_colors[:len(paths)], paths):
        for i in range(len(path) - 1):  # Going over each edge in the path.
            add_edge(G_multi, path[i], path[i + 1], darken_color(color))
    return G_multi


def visualizeGraph(G, pos, num, weights, ax=None):
    nx.draw_networkx_nodes(G, pos, list(range(num)), node_shape='o',
                           edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, width=0.5, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights, font_size=8,
                                 font_family="serif", rotate=False, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="serif", ax=ax)


def visualizeMultigraph(G_multi, pos, num_nodes, ax=None):
    nx.draw_networkx_nodes(G_multi, pos, list(range(num_nodes)), node_shape='o',
                           edgecolors="black", linewidths=0.5, ax=ax)
    for edge in G_multi.edges(data=True):
        nx.draw_networkx_edges(G_multi, pos,
                               edgelist=[(edge[0], edge[1])],
                               connectionstyle=f'arc3, rad = {edge[2]["rad"]}',
                               width=1.5, arrowstyle="-",
                               edge_color=edge[2]["color"], ax=ax)
    nx.draw_networkx_labels(G_multi, pos, font_size=6, font_family="serif", ax=ax)


def createPaths(adj):
    paths = []

    for i in range(len(adj[0]) - 1):
        paths.extend([(i, x) for x in range(i + 1, len(adj[0])) if adj[i][x] > 0])

    return paths


def runSimulation(start, end):
    G = createGraph(4)
    pos = nx.spring_layout(G)

    count = start
    adj = transform(0)
    adj_list = []

    for x in range(len(adj) * len(adj[0])):
        adj_list.append([])

    for i in range(end - start):
        adj = transform(count)

        for x in range(len(adj) * len(adj[0])):
            adj_list[x].append(adj[math.floor(x / len(adj[0]))][x % len(adj[0])])

        plt.figure()
        G_multi = createMultigraph(4, createPaths(adj))
        visualizeMultigraph(G_multi, pos, 4)
        plt.tight_layout()
        plt.savefig(f'output/output{count}.png')
        plt.close()

        count = count + 1

    return adj_list


def list_to_matrix(adj):
    ret = []

    size = int(math.sqrt(len(adj)))

    for i in range(len(adj[0])):
        curr = []
        for x in range(size):
            group = []

            for y in range(size):
                group.append(adj[size * x + y][i])

            curr.append(group)

        ret.append(curr)

    return ret


def to_matrix(l, n):
    return [l[i:i + n] for i in xrange(0, len(l), n)]


def get_std_dev(adj, pred):
    return math.sqrt(sum([math.pow(adj[i] - pred[i], 2) for i in range(len(adj))]) / (len(pred) - 1))


def zTest(adj, pred, stddev):
    flagged = []

    for i in range(len(adj)):
        flag = []

        for x in range(len(adj[i])):
            flag.append(abs(adj[i][x] - pred[i][x]) > 4 * stddev[int(math.sqrt(len(adj))) * i + x])

        flagged.append(flag)

    return flagged


def countErrors(choices, outliers, flagged):
    choices.append(sys.maxsize)
    errors = 0
    choice = 0
    for count, flag in enumerate(flagged):
        temp = choice

        for i in range(len(flag)):
            for x in range(len(flag[i])):
                if count < choices[choice]:
                    if flag[i][x]:
                        errors = errors + 1

                if count == choices[choice]:
                    if (i, x) == outliers[choice] and not flag[i][x]:
                        errors = errors + 1

                    elif (i, x) != outliers[choice] and flag[i][x]:
                        errors = errors + 1

                    temp = choice + 1

        choice = temp

    return errors


def evaluateModel(history, config_tuples, start, end):
    test_list = runSimulation(start, end)

    [noise(x) for x in test_list]

    choices = sorted(random.sample(list(range(0, end - start)), k=int((end - start) / 4)))

    size = int(math.sqrt(len(history)))

    stddev = []
    lists = []
    for i in range(size * size):
        model = SARIMAX(history[i], order=config_tuples[math.floor(i / size)][i % size][0],
                        seasonal_order=config_tuples[math.floor(i / size)][i % size][1],
                        trend=config_tuples[math.floor(i / size)][i % size][2], enforce_stationarity=False,
                        enforce_invertibility=False)
        # fit model
        model_fit = model.fit(disp=False)

        predictions = model_fit.predict(start, end)

        print("asd")
        print(predictions)
        print(test_list[i])
        print(get_std_dev(test_list[i], predictions))

        stddev.append(get_std_dev(test_list[i], predictions))

        lists.append(predictions)

    test = list_to_matrix(test_list)
    predictions = list_to_matrix(lists)

    flagged = []

    for i in range(len(test)):
        flagged.append(zTest(test[i], predictions[i], stddev))

    outliers = outlier(test_list, choices)

    return countErrors(choices, outliers, flagged)


adj_list = runSimulation(0, 50)

gs = GridSearchSARIMA()

# data split
n_test = 25  # model configs
cfg_list = gs.sarima_configs(seasonal=[0, 2])

scores = []

#for i in range(len(adj_list)):
#    scores.append(gs.grid_search(adj_list[i], cfg_list, n_test, False)[0])

#print(scores)

# tuple(map(int, test_str.split(', ')))

config_tuples = [[[(0, 0, 0), (0, 0, 0, 0), 'n'], [(1, 0, 0), (1, 0, 0, 2), 'n'], [(1, 0, 1), (0, 0, 0, 0), 'ct']], [[(0, 0, 0), (0, 0, 0, 0), 'n'], [(0, 0, 0), (0, 0, 0, 0), 'n'], [(1, 0, 0), (1, 0, 0, 2), 'n']], [[(0, 0, 0), (0, 0, 0, 0), 'n'], [(0, 0, 0), (0, 0, 0, 0), 'n'], [(0, 0, 0), (0, 0, 0, 0), 'n']]]


print(config_tuples)

print(evaluateModel(adj_list, config_tuples, 50, 100))
