import numpy as np
import pandas as pd
import datetime
from city import City
from utility import get_direction, get_haversine_distance
import operator
import pickle
import random
from tqdm import tqdm
from distance import distance_matrix
import matplotlib.pyplot as plt

class GA:

    def __init__(self,
                points,
                distance_matrix,
                temperature,
                humidity,
                wind_speed,
                wind_direction,
                initial_departure_time,
                model,
                pca,
                pop_size=500, 
                elite_size=100, 
                mutation_rate=0.05, 
                generations=60
                ):

        self.distance_matrix = distance_matrix
        self.temperature = temperature
        self.humidity = humidity
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.points = points
        self.idx = range(1,len(points))
        self.model = model
        self.initial_departure_time = initial_departure_time
        self.weekday = 1  if initial_departure_time.weekday() < 5 else 0
        self.dayofweek = initial_departure_time.weekday()
        self.pca = pca
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.track_best_score = []
        self.track_best = []

    def createRoute(self):
        route = random.sample(self.idx, len(self.points)-1)
        return route

    def initialPopulation(self):
        population = []
        for i in range(0, self.pop_size):
            population.append(self.createRoute())
        return population
    def fitness(self,route):

        path_time = 0
        departure_time = self.initial_departure_time

        def get_time(city,to_city,idx,to_city_idx):

            inputs = [city.longitude, city.latitude,to_city.longitude,to_city.latitude,departure_time.day]
            pickup_time = departure_time.hour + departure_time.minute/60
            displacement = get_haversine_distance(city.latitude,city.longitude,to_city.latitude,to_city.longitude)
            total_distance = self.distance_matrix[idx,to_city_idx]
            direction = get_direction(city.latitude,city.longitude,to_city.latitude,to_city.longitude)
            tmp_dt = departure_time.replace(second=0,minute = 0,microsecond =0)
            temperature = self.temperature.loc[self.temperature['datetime'] == tmp_dt,'temperature'].values[0]
            humidity = self.humidity.loc[self.humidity['datetime'] == tmp_dt,'humidity'].values[0]
            wind_speed = self.wind_speed.loc[self.wind_speed['datetime'] == tmp_dt,'wind_speed'].values[0]
            wind_direction = self.wind_direction.loc[self.wind_direction['datetime'] == tmp_dt,'wind_direction'].values[0]
            
            pickup_transform = self.pca.transform([[city.latitude,city.longitude]])
            pickup_pca0 = pickup_transform[:, 0][0]
            pickup_pca1 = pickup_transform[:, 1][0]
            dropoff_transform = self.pca.transform([[to_city.latitude,to_city.longitude]])
            dropoff_pca0 = dropoff_transform[:, 0][0]
            dropoff_pca1 = dropoff_transform[:, 1][0]
            inputs.extend([pickup_time,self.dayofweek,self.weekday,displacement,direction,temperature,humidity,wind_speed,
                            wind_direction,pickup_pca0,pickup_pca1,dropoff_pca0,dropoff_pca1])


            time_taken = np.exp(self.model.predict([inputs])[0]) - 1
            return time_taken
        time_taken =get_time(self.points[0],self.points[route[0]],0,route[0])
        added_seconds = datetime.timedelta(0, time_taken)
        departure_time += added_seconds #to change
        path_time += time_taken
        
        for i,idx in enumerate(route):
            city = self.points[idx]
            to_city = None

            if i + 1 < len(route):
                to_city = self.points[route[i+1]]
                to_city_idx = i+1
            else:
                to_city = self.points[route[0]]
                to_city_idx = 0
            
            
            time_taken = get_time(city,to_city,idx,to_city_idx)
            added_seconds = datetime.timedelta(0, time_taken)
            departure_time += added_seconds #to change
            path_time += time_taken
        try:
            fitness = 1/path_time
        except:
            print(route)
        return fitness

    #Determine fitness
    def rankRoutes(self,population):
        #population is a list of route
        #returns dictionnary {route ID: fitness}sorted by fitness (1/time)
        fitnessResults = {}
        for i in tqdm(range(0,len(population))):
            fitnessResults[i] = self.fitness(population[i])
        return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)
    
    #Selecting mating pool
    def selection(self,popRanked):
        selectionResults = []
        df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
        df['cum_sum'] = df.Fitness.cumsum()
        df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()
        
        for i in range(0, self.elite_size):
            selectionResults.append(popRanked[i][0])
        for i in range(0, len(popRanked) - self.elite_size):
            pick = 100*random.random()
            for i in range(0, len(popRanked)):
                if pick <= df.iat[i,3]:
                    selectionResults.append(popRanked[i][0])
                    break
        return selectionResults
    
    #Mating pool = returns subsets of population according to selection results indices returned previously
    def matingPool(self,population, selectionResults):
        matingpool = []
        for i in range(0, len(selectionResults)):
            index = selectionResults[i]
            matingpool.append(population[index])
        return matingpool
    
    def crossover(self,parent1, parent2):

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

        pos=random.randrange(1,len(parent1)-1)
        child1 = parent1[:pos] + parent2[pos:] 
        child2 = parent2[:pos] + parent1[pos:] 
        
        return  process_gen_repeated(child1, child2)


    def crossParentsPopulation(self,parents):
        childs=[]
        for i in range(0,len(parents),2):
            childs.extend(self.crossover(parents[i],parents[i+1]))
        return childs


    #Mutate
    def inversion_mutation(self,chromosome_aux):
        chromosome = chromosome_aux     
        index1 = random.randrange(0,len(chromosome))
        index2 = random.randrange(index1,len(chromosome))
                    
        chromosome_mid = chromosome[index1:index2]
        chromosome_mid.reverse()         
        chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
                    
        return chromosome_result
    
    def mutation(self, population):
            
        def inversion_mutation(chromosome_aux):
            chromosome = chromosome_aux
            
            index1 = random.randrange(0,len(chromosome))
            index2 = random.randrange(index1,len(chromosome))
            
            chromosome_mid = chromosome[index1:index2]
            chromosome_mid.reverse()
            
            chromosome_result = chromosome[0:index1] + chromosome_mid + chromosome[index2:]
            
            return chromosome_result
    
        aux = []
        for i in range(len(population)):
            if random.random() < self.mutation_rate :
                aux.append(inversion_mutation(population[i]))
            else:
                aux.append(population[i])
        return aux

    #Next generation
    def nextGeneration(self,currentGen):
        popRanked = self.rankRoutes(currentGen)
        self.track_best_score.append(1/popRanked[0][1])
        tmp = list(currentGen[popRanked[0][0]])
        tmp.insert(0,0)
        tmp.append(0)
        self.track_best.append(tmp)
        selectionResults = self.selection(popRanked)
        matingpool = self.matingPool(currentGen, selectionResults)
        children = self.crossParentsPopulation(matingpool)
        nextGeneration = self.mutation(children)
        return nextGeneration
    
    def run(self):
        pop = self.initialPopulation()
        init = self.rankRoutes(pop)
        print("Initial time: " + str(1 / init[0][1]))
        self.track_best_score.append(1 / init[0][1])
        tmp = list(pop[init[0][0]])
        tmp.insert(0,0)
        tmp.append(0)
        self.track_best.append(tmp)
        for _ in tqdm(range(0, self.generations)):
            pop = self.nextGeneration(pop)
        print("Final time: " + str(1 / self.rankRoutes(pop)[0][1]))
        bestRouteIndex = self.rankRoutes(pop)[0][0]
        bestRoute = pop[bestRouteIndex]
        bestRoute.insert(0,0)
        bestRoute.append(0)
        plt.plot(self.track_best_score)
        plt.savefig('training.png')
        print(self.track_best)
        np.save('gif.npy',np.array(self.track_best))
        return bestRoute


cities = []
with open('../data/example.txt', 'r') as f:
    lines = f.readlines()
    lines = [lines.strip().split() for lines in lines]
    for line in lines:
        city = City(float(line[0]),float(line[1]))
        cities.append(city)
distance_matrix = np.load('distance_matrix_50.npy')
# np.save('distance_matrix_50.npy',distance_matrix)
# print(distance_matrix)

weather = ['temperature','humidity','wind_speed','wind_direction']
wl = []
for element in weather:
    tmp  = pd.read_csv(f'../data/{element}.csv',sep = ',',nrows=40000)
    tmp['datetime'] = pd.to_datetime(tmp['datetime'])
    tmp = tmp[['datetime', 'New York']].rename(columns = {'New York' : element})
    wl.append(tmp)

model = pickle.load(open('../data/finalized_model.sav','rb'))
pca = pickle.load(open('../data/pca_model.sav','rb'))

initial_departure_time = datetime.datetime(2016, 5, 10,8)
test = GA(cities,distance_matrix,wl[0],wl[1],wl[2],wl[3],initial_departure_time,model,pca)
print(test.run())