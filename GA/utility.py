
import random
import numpy as np

def fitness(route,model,initial_departure_time):

    path_time = 0
    departure_time = initial_departure_time
    for i,city in enumerate(route):
        to_city = None
        if i + 1 < len(route):
            to_city = route[i+1]
        else:
            to_city = route[0]
        


        time_taken = city.time(to_city,departure_time, model)
        departure_time += time_taken #to change
        path_time += time_taken
    fitness = 1/path_time
    return fitness

def createRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(createRoute(cityList))

    return population

def get_haversine_distance(lat_1, long_1, lat_2, long_2):
    """
    Calculate the distance of 2 points with consideration of the roundness of earth.
    """
    
    AVG_EARTH_RADIUS = 6371
    lat_1, long1, lat_2, long_2 = map(np.radians, (lat_1, long_1, lat_2, long_2))
    lat = lat_2 - lat_1 ; long = long_2 - long_1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin(long * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def get_direction(lat_1, long_1, lat_2, long_2):
    """
    Calculates the angle or direction of 2 points with consideration of the roundness of earth.
    """
    
    AVG_EARTH_RADIUS = 6371  # in km
    long_delta_rad = np.radians(long_2 - long_1)
    lat_1, long_1, lat_2, long_2 = map(np.radians, (lat_1, long_1, lat_2, long_2))
    y = np.sin(long_delta_rad) * np.cos(lat_2)
    x = np.cos(lat_1) * np.sin(lat_2) - np.sin(lat_1) * np.cos(lat_2) * np.cos(long_delta_rad)
    
    return np.degrees(np.arctan2(y, x))