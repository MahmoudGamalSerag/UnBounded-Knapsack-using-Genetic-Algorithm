import random
import sys
import operator
from typing import List, Any


class Knapsack(object):

    def __init__(self):

        self.C = 0
        self.weights = []
        self.profits = []
        self.opt = []
        self.parents = []
        self.newparents = []
        self.bests = []
        self.best_p = []
        self.iterated = 1
        self.population = 0
        self.population1 = []
        self.population1_b = []
        self.opt1 = [0, []]
        self.b_opt = [0, []]
        self.parents1= []


    def initialize(self):

        for i in range(self.population):
            parent = []
            for k in range(0, len(weights)):
                k = random.uniform(0, 1)
                parent.append(k)
            self.parents.append(parent)
            self.parents1 = self.parents

    def properties(self, weights, profits, C, population):

        self.weights = weights
        self.profits = profits
        self.C = C
        self.population = population
        self.initialize()

    def fitness(self, item):

        sum_w = 0
        sum_p = 0
        i = 0.0


        for index, i in enumerate(item):
            if i == 0.0:
                continue
            elif i>0.0:
                sum_w = sum_w + (self.weights[index]*i)
                sum_p = sum_p + (self.profits[index]*i)

        if sum_w > self.C:
            return -1
        else:
            return sum_p

    def evaluation(self):

        best_pop = self.population
        for i in range(len(self.parents)):
            parent = self.parents[i]
            ft = self.fitness(parent)
            self.bests.append((ft, parent))


        self.bests.sort(key=operator.itemgetter(0), reverse=True)
        self.best_p = self.bests[:best_pop]
        self.best_p = [x[1] for x in self.best_p]


    def mutation(self, ch , mutation):

        for i in range(len(ch)):
            k = random.uniform(0, 1)
            if k > mutation:
                ch[i]=random.uniform(0,1)
        return ch


    def crossover(self, ch1, ch2):
        threshold = random.randint(1, len(ch1)-1)
        tmp1 = ch1[threshold:]
        tmp2 = ch2[threshold:]
        ch1 = ch1[:threshold]
        ch2 = ch2[:threshold]
        ch1.extend(tmp2)
        ch2.extend(tmp1)

        return ch1, ch2

    def run(self):

        self.evaluation()
        newparents = []
        pop = len(self.best_p)-1

        sample = random.sample(range(pop), pop)
        for i in range(0, pop):

            if i < pop-1:
                r1 = self.best_p[i]
                r2 = self.best_p[i+1]
                nchild1, nchild2 = self.crossover(r1, r2)
                newparents.append(nchild1)
                newparents.append(nchild2)
            else:
                r1 = self.best_p[i]
                r2 = self.best_p[0]
                nchild1, nchild2 = self.crossover(r1, r2)
                newparents.append(nchild1)
                newparents.append(nchild2)

        for i in range(len(newparents)):
            pop_best = len(newparents)
            newparents[i] = self.mutation(newparents[i], mutation)
            parent1 =newparents[i]
            ft = self.fitness(parent1)
            self.population1.append((ft, parent1))
            self.population1.sort(key=operator.itemgetter(0), reverse=True)
            self.population1_b = self.population1[:pop_best]
            self.population1_b = [x[1] for x in self.population1_b]
        print(self.population1)

        self.opt1 = self.population1[0]
        print(self.opt1)
        op = self.opt1[0]
        op1 = self.b_opt[0]
        if op > op1:
            self.b_opt = self.opt1

        if self.iterated<100:
            self.iterated += 1
            print("recreate generations for {} time".format(self.iterated))
            self.parents = newparents
            self.bests = []
            self.best_p = []
            self.opt1 = [0, []]
            self.run()
        else:
            print("the best result found in all generations is: {}" .format(self.b_opt))


weights = []
profits = []
C = int(input("enter capacity: "))
n = int(input("Enter number of elements : "))
population = int(input("Enter desired population number: "))
mutation = float(input("Enter desired mutation percentage number from 0 to 1: "))
for i in range(0, n):
    w = int(input("Enter weight of {} element : ".format(i+1)))
    p = int(input("Enter profit of {} element : ".format(i+1)))
    weights.append(w)
    profits.append(p)


k = Knapsack()
k.properties(weights, profits, C, population)
k.run()