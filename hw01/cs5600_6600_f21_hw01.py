
####################################################
# CS 5600/6600: F20: Assignment 1
# Carson Fox
# A02251670
#####################################################

import numpy as np

def binary_activation(x, weights, bias):
    return 1 if x @ weights + bias > 0 else 0


class and_percep:
    def __init__(self):
        self.b = -2
        self.w1 = np.array([[1.1], [1.1]])

    def output(self, x):
        return binary_activation(x, self.w1, self.b)


class or_percep:
    def __init__(self):
        self.b = -1
        self.w1 = np.array([[1.1], [1.1]])

    def output(self, x):
        return binary_activation(x, self.w1, self.b)


class not_percep:
    def __init__(self):
        self.b = 1
        self.w1 = np.array([[-1.1]])

    def output(self, x):
        return binary_activation(x, self.w1, self.b)


class xor_percep:
    def __init__(self):
        self.and_per = and_percep()
        self.or_per = or_percep()
        self.not_per = not_percep()

    def output(self, x):
        # [x0 & x1, x0 | x1]
        y1 = np.array([self.and_per.output(x), self.or_per.output(x)])

        # [!(x0 & x1), x0 | x1]
        y2 = np.array([self.not_per.output(y1[0:1]), y1[1]])

        # !(x0 & x1) & x0 | x1 = x0 ^ x1
        return self.and_per.output(y2)


def binary_feed_forward(x, weights, biases):
    y = x @ weights + biases
    activation = np.frompyfunc(lambda y: 1 if y > 0 else 0, 1, 1)
    return activation(y)


class xor_percep2:
    def __init__(self):
        # And and or perceptrons
        self.w1 = np.array([[1.1, 1.1], [1.1, 1.1]])
        self.b1 = np.array([-2, -1])

        #Combine the two, nand overrides or
        self.w2 = np.array([[-1.2], [1.1]])
        self.b2 = np.array([-1])

    def output(self, x):
        y1 = binary_feed_forward(x, self.w1, self.b1)
        y2 = binary_feed_forward(y1, self.w2, self.b2)
        return y2


class percep_net:
    def __init__(self):
        self.and_per = and_percep()
        self.or_per = or_percep()
        self.not_per = not_percep()


    def output(self, x):
        x0_or_x1 = self.or_per.output(x[0:2])
        not_x2 = self.not_per.output(x[2:3])
        or_and_not = self.and_per.output(np.array([x0_or_x1, not_x2]))
        return self.or_per.output(np.array([or_and_not, x[3]]))
