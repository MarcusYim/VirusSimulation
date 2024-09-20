import random
from itertools import chain


class DataCenterSimulation:
    def __init__(self):
        self.adj = [[0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0]]

        self.activations = [[0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0]]

        self.front_functions = [lambda x: 1 + (random.random() - 0.5), lambda x: 1 + (random.random() - 0.5), lambda x: 1 + (random.random() - 0.5)]

        self.front = [0, 1, 2]
        self.middle = [3, 4, 5]
        self.back = [6, 7, 8]
        self.front_count = 0
        self.count = 1

    def front_activation(self):
        for count, index in enumerate(self.front):
            for count, edge in enumerate(self.adj[index]):
                if edge > 0:
                    self.activations[index][count] = self.front_functions[index](self.front_count)

        self.front_count += 1

    def activate_all_children(self, index, sum_activations):
        print(self.activations[index])

        for count, edge in enumerate(self.adj[index]):
            if edge > 0:
                self.activations[index][count] = sum_activations

    def middle_activation(self):
        for index in self.middle:
            activation_count = 0
            activation_sum = 0

            for list in self.activations:
                if list[index] > 0:
                    activation_count += 1
                    activation_sum += list[index]

                list[index] = 0

            if activation_count % 2 == 1:
                self.activate_all_children(index, activation_sum)

    def back_activation(self):
        for index in self.back:
            for list in self.activations:
                list[index] = 0

    def step_forward(self):
        if self.count == 1:
            self.front_activation()

        elif self.count % 2 == 0:
            self.middle_activation()

        elif self.count % 3 == 0:
            self.back_activation()
            self.count = 1
            return

        self.count += 1

if __name__ == '__main__':
    dc = DataCenterSimulation()

    print(len(dc.activations))

    f = open("data.csv", "w")

    f.write(str(list(range(0, len(dc.activations) * len(dc.activations)))).replace("[", "").replace("]", "").replace(" ", "") + "\n")

    for i in range(200):
        dc.step_forward()
        #print(len(dc.activations) * len(dc.activations) - 1)
        f.write(list(chain.from_iterable(dc.activations)).__str__().replace("[", "").replace("]", "").replace(" ", "") + "\n")
        #f.write(str(list([0 for i in range(0, len(dc.activations) * len(dc.activations))])).replace("[", "").replace("]", "") + "\n")
        #f.write(str(list([i for x in range(0, len(dc.activations) * len(dc.activations))])).replace("[", "").replace("]", "") + "\n")

    f.close()

    #print(dc.activations)
