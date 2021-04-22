import numpy as np
import requests
import json
import pandas as pd
import time
from tqdm import tqdm
from city import City

# does an api call to compute distance between two points (manhattan)
def distance(lat_i, long_i, lat_j, long_j):
    # in meters
    # give long, lat of coordiate 1 then long, lat of coordinate 2
    base_url = "http://router.project-osrm.org/route/v1/driving/{},{};{},{}"
    url = base_url.format(lat_i, long_i, lat_j, long_j)
    # print(url)
    done = False
    while not done:
        res = requests.get(url)
        try:
            res_dict = json.loads(res.text)
            done = True
        except:
            print('hey')
            time.sleep(0.1)
    return res_dict["routes"][0]["distance"]

def distance_matrix(cityList):
    n= len(cityList)
    res = np.zeros((n,n))
    for i in tqdm(range(n)):
        for j in range(n):
            if i == j:
                continue
            else:
                print(j)
                start, end = cityList[i],cityList[j]
                res[i][j] = distance(start.longitude, start.latitude, end.longitude, end.latitude)
    return res
# cities = []
# with open('../data/example.txt', 'r') as f:
#     lines = f.readlines()
#     lines = [lines.strip().split() for lines in lines]
#     for line in lines:
#         city = City(float(line[0]),float(line[1]))
#         cities.append(city)
#     print(cities)
# np.save('distance_matrix.npy',distance_matrix(cities))
# print(distance(-73.98041534423827,40.738563537597656,-73.99948120117188,40.731151580810554))
