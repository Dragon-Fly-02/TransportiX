from .utility import fitness 
from random import *
import numpy as np
import pandas as pd
import operator

#Determine fitness
def rankRoutes(population,model, initial_departure_time):
#population is a list of route
#returns dictionnary {route ID: fitness}sorted by fitness (1/time)
    fitnessResults = {}
    for i in range(0,len(population)):
        fitnessResults[i] = fitness(population[i],model,initial_departure_time)
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


#Selecting mating pool
def selection(popRanked,nbSelectedForMating):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
    
    for i in range(0, nbSelectedForMating):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - nbSelectedForMating):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

#Mating pool = returns subsets of population according to selection results indices returned previously
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool



def crossover(parent1,parent2):     
    pos=random.randrange(1,len(parent1)-1) #check len
    child1 = parent1[:pos] + parent2[pos:] 
    child2 = parent2[:pos] + parent1[pos:]   
    #Cross-over or breeding 
    def process_gen_repeated(copy_child1,copy_child2):
        count1=0
        for gen1 in copy_child1[:pos]:
            repeat = 0
            repeat = copy_child1.count(gen1)
            if repeat > 1:#If need to fix repeated gen
                count2=0
                for gen2 in parent1[pos:]:#Choose next available gen
                    if gen2 not in copy_child1:
                        child1[count1] = parent1[pos:][count2]
                    count2+=1
                count1+=1

        count1=0
        for gen1 in copy_child2[:pos]:
            repeat = 0
            repeat = copy_child2.count(gen1)
            if repeat > 1:#If need to fix repeated gen
                count2=0
                for gen2 in parent2[pos:]:#Choose next available gen
                    if gen2 not in copy_child2:
                        child2[count1] = parent2[pos:][count2]
                    count2+=1
            count1+=1
        return [child1,child2]

    return  process_gen_repeated(child1, child2)


def crossParentsPopulation(parents):
    childs=[]
    for i in range(0,len(parents),2):
        childs.append(crossover(parents[i],parents[i+1]))
    return childs


#Mutate
def inversion_mutation(chromosome_aux):
    chromosome = chromosome_aux     
    index1 = randrange(0,len(chromosome))
    index2 = randrange(index1,len(chromosome))
                
    chromosome_mid = chromosome[index1:index2]
    chromosome_mid.reverse()         
    chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
                
    return chromosome_result
 
def mutate(individual,mutationRate):
    mutatedIndividual = []
    for _ in range(len(individual)):
        if random.random() < prob :
            mutatedIndividual = inversion_mutation(individual)
    return mutatedIndividual

def mutatePopulation(population, mutationRate):
    mutatedPop = []
    for ind in range(0, len(population)):
        mutatedIndividual = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedIndividual)
    return mutatedPop

#Next generation
def nextGeneration(currentGen, eliteSize, mutationRate, model, initial_departure_time):
    popRanked = rankRoutes(currentGen,model, initial_departure_time)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = crossParentsPopulation(matingpool)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

