#converts html into png

from GrabzIt import GrabzItImageOptions
from GrabzIt import GrabzItClient

def html_str(file):
    res = ""
    with open(file) as infile:
        for l in infile:
            res += f"{l}"
    return res
  
def png_map(latlon):
    # example input latlon = [(40.76372909545898, -73.97139739990234), 
    #      (40.77077865600585, -73.95086669921875), 
    #      (40.722164154052734, -73.99718475341797)]
    NYC_map = folium.Map(location = [40.7128, -74.0160], zoom_start = 11)
    for coord in latlon:
        folium.Marker( location=[ coord[0], coord[1] ], fill_color='#43d9de', radius=8 ).add_to( NYC_map )
    NYC_map.save('NYC.html')
    file = html_str('NYC.html')
    
    grabzIt = GrabzItClient.GrabzItClient("NjdmOGEzMTU2MTk2NGJlZWI5ODkyNTUwYmQyOWY3MWQ=", "UkFIP10/Pz8/QT8/Pzc/bD97Pz89OGU/CT9UYD8/P2M=")
    options = GrabzItImageOptions.GrabzItImageOptions()
    options.format = "png"
    grabzIt.HTMLToImage(file, options)
    grabzIt.SaveTo("NYC_map.png")
    
