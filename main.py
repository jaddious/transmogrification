from transmogrification import * 
import geopandas as gpd
import matplotlib.pyplot as plt
import os
from shapely.geometry import Polygon
from shapely.ops import transform

boundaries = "data/boundaries.shp"
boundaries = gpd.read_file(boundaries)
# get name == "centerline"
centerline = boundaries[boundaries['name'] == 'center'].geometry.values[0]
top_line = boundaries[boundaries['name'] == 'top'].geometry.values[0]
bottom_line = boundaries[boundaries['name'] == 'bottom'].geometry.values[0]

corridor = Polygon(list(top_line.coords) + list(bottom_line.coords)[::-1])

#layers = ["data/layers/rivers.shp", "data/layers/ROI.shp"]
#use same EPSG for all layers and don't include Z values

"""layers = ["data/layers/ROI.shp", "data/layers/rivers.shp",
          "data/layers/topography.shp", "data/layers/trip.shp", 
          "data/layers/sea.shp", "data/layers/urban.shp", "data/layers/urban.shp", 
          "data/layers/grid_50.shp", "data/layers/images.shp", "data/layers/NUTS.shp", "data/layers/CEMT.shp", "data/layers/grid_250.shp"]"""
          
          
layers = [r"G:\.shortcut-targets-by-id\1xXF4_DHb79MFlyi9vLdZcN_SagP0sxVr\Academie Beaux Arts\01-GIS\01-Layers\TENtec\inland_waterways.shp"]

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].set_title("Original geometries")
axs[1].set_title("Straightened geometries (s,n)")

def display_layer(ax, data, type, color='blue'): 
    if type == "Polygon":
        # display the polygons in ax 1
        data.boundary.plot(ax=ax, color=color)
    elif type == "MultiPolygon":
        data.boundary.plot(ax=ax, color=color)
    elif type == "LineString":
        data.plot(ax=ax, color=color)
    elif type == "Point":
        data.plot(ax=ax, color=color, markersize=5)

display_layer(axs[0], gpd.GeoDataFrame(geometry=[centerline]), "LineString", "black")
display_layer(axs[0], gpd.GeoDataFrame(geometry=[top_line]), "LineString", "yellow")
display_layer(axs[0], gpd.GeoDataFrame(geometry=[bottom_line]), "LineString", "orange")


# Choose a target rectangle (optional). Comment out if you want raw UV in [0,1].
RECT = (0, 0, 1, 1)  # e.g., image space width=1000, height=600

for layer in layers: 
    print(f"Processing layer: {layer}")
    data = gpd.read_file(layer)
    if data.crs.to_epsg() != 4327:
        data = data.to_crs(epsg=4327)
    display_layer(axs[0], data, data.geometry.geom_type.unique()[0])
    # clip to corridor
    
    # print all types in the layer
    
    # simpleify geometry to speed up processing
    data.geometry = data.geometry.simplify(
        tolerance=0.005, preserve_topology=True)
    data = gpd.clip(data, corridor)
    # multipart to singlepart
    data = data.explode(index_parts=False)
    print("Geometry types in layer:", data.geometry.geom_type.unique())
    display_layer(axs[0], data, data.geometry.geom_type.unique()[0], "red")
    type = data.geometry.geom_type.unique()[0]
    
    print("THere are", len(data), "features in this layer")
    # convert to EPSG:4327

    data.geometry = straighten(data.geometry.values, centerline=centerline, top_line=top_line, bottom_line=bottom_line)
    
    # only keep valid geometries
    
    #move the straightened geometry so that the origin is at (0,0)
    data = data[~data.is_empty & data.geometry.notnull()]
    # only keep type matching geometries
    data = data[data.geometry.geom_type == type]
    
    print("After straightening, there are", len(data), "features in this layer")
    basename = os.path.basename(layer).split(".")[0]
    output_path = f"data/transformed/{basename}_straightened.shp"

    data.to_file(output_path)
    print("final geom count:", len(data))
    display_layer(axs[1], data, type)
    

plt.show()
    
    