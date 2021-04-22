

class City:

    def __init__(self,longitude,latitude):

        self.longitude = longitude
        self.latitude = latitude
    
    def time(self,arrived_city,departure_time,model):
        """
        Return time taken between self and another city
        """

        inputs = [self.longitude,self.latitude,arrived_city.latitude,arrived_city.longitude,departure_time]
        #add model
        time = model.predict(inputs)[0]
        return time
    
    def __eq__(self,other):
        return self.longitude == other.longitude and self.latitude == other.latitude
    
    def __repr__(self):
        return "(" + str(self.longitude) + "," + str(self.latitude) + ")"
    
