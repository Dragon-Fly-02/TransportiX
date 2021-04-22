from .vrp import *
from .utility import *
from .city import *


def geneticAlgorithm(cityList, popSize, eliteSize, mutationRate, generations,model,initial_departure_time):
    pop = initialPopulation(popSize, cityList)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate, model, initial_departure_time)
    
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

cityList = []#complete
bestRoute = geneticAlgorithm(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=500) #vary hyperparameters ? 
print(bestRoute)
