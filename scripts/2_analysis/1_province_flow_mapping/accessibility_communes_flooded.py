# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:22:01 2018

@author: cenv0574
"""

import geopandas as gpd
import pandas as pd
import os
import networkx as nx
import numpy as np
import json

from geopy.distance import vincenty
from boltons.iterutils import pairwise


def main():

    # Define current directory and data directory
    config_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), '..', '..','..', 'config.json')
    )
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    data_path = config['paths']['data']
    calc_path = config['paths']['calc']
    output_path = config['paths']['output']

    # load commune geospatial dataset
    commune_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','commune_level_stats.shp')
    communes = gpd.read_file(commune_path)
    
    # load provinces
    province_path = os.path.join(data_path,'Vietnam_boundaries','who_boundaries','who_provinces.shp')
    provinces = gpd.read_file(province_path)
    Thanh_Hoa = provinces.loc[provinces.NAME_ENG == 'Thanh Hoa']
    
    # load communes Thanh Hoa and clean the dataframe a bit
    communes_TH = communes.loc[communes.pro_name_e == 'Thanh Hoa']
    communes_TH = communes_TH.drop(['gid', 'objectid','who_provin', 'prov_ec_id',
       'who_distri', 'dis_ec_id', 'name_vie', 'name_eng', 'dis_name_v',
       'dis_com_co', 'commune_ec', 'dis_name_e', 'pro_name_v','address', 'unnamed_ 1', 'unnamed__1','shape_leng', 'shape_le_1', 'shape_area'], axis=1)    

    # Load places from OSM
    country_pbf = os.path.join(data_path,'OSM','vietnam-latest.osm.pbf')
    place_path = os.path.join(data_path,'OSM','places.shp')
    
    if os.path.isfile(place_path) is not True:
        os.system("ogr2ogr -progress -overwrite -f \"ESRI Shapefile\" -sql \
                   \"SELECT place FROM points WHERE place is not NULL\" \
                   -lco ENCODING=UTF-8 -skipfailures "+place_path+" "+country_pbf)    
    
    places = gpd.read_file(place_path)
    places.geometry = places.centroid
        
    # clip places to Thanh Hoa (pretty efficient code if I may say so myself)
    places_TH = places.loc[places['geometry'].apply(lambda x: x.within(Thanh_Hoa['geometry'].values[0]))]
    
    # only keep towns and bigger:
    places_TH = places_TH.loc[places_TH.place.isin(['town','city'])]
    
    places_TH.to_file(os.path.join(data_path,'OSM','places_TH.shp'))
    
    # create spatial index of towns and bigger file
    spatial_places = places_TH.sindex
    
    # Read road network for Thanh Hoa
    road_path = os.path.join(data_path,'Roads','ThanhHao_roads','road_thanhhaoedges.shp')
    roads_TH = gpd.read_file(road_path)

    if 'length' not in roads_TH:
        roads_TH['length'] = roads_TH.geometry.apply(line_length)
        speed_dict = dict(zip(list(roads_TH['class'].unique()),[40,60,40,80]))
        roads_TH['speed'] = roads_TH['class'].apply(lambda x: speed_dict[x])
        roads_TH['t_time'] = roads_TH['length'].div(roads_TH['speed'])
        roads_TH.to_file(road_path)
    
    g = nx.read_shp(road_path)
    sg = max(nx.connected_component_subgraphs(g.to_undirected()), key=len)

    nodes = np.array(sg.nodes())

    # look up nearest node for each centroid of commune and place

    # for communes
    communes_TH['nearest_node'] = communes_TH.geometry.apply(lambda x: find_node(x,nodes))
     
    #for places
    places_TH['nearest_node'] = places_TH.geometry.apply(lambda x: find_node(x,nodes))


    rps = ['no_flood','1_freq','5_freq','10_freq','20_freq']

    # # Save flooded networks
    save_flooded_networks(rps,roads_TH,calc_path,output_path)
    
    #   # estimate distance to nearest place for each commune centroid
    distance_per_rp = {}
    for rp in rps:
        distance_per_rp[rp] = shortest_distance(rp,calc_path,road_path,communes_TH,places_TH,spatial_places)
        
    # merge with initial commune dataset
    new_distances = pd.concat(distance_per_rp,axis=1)
    new_distances.columns = [x[0] for x in list(new_distances.columns)]
    communes_affected = communes_TH.merge(new_distances, left_on='commune_id',right_index=True)    
    communes_affected = communes_affected.drop('nearest_node',axis=1)

    # # save output 
    communes_affected.to_file(os.path.join(output_path,'Thanh_Hoa_affected_communes_flood.shp'))


def find_node(x,nodes):
    centroidal = list(x.centroid.coords)[:1][0]
    return nodes[np.argmin(np.sum((nodes - centroidal)**2, axis=1))]   


def save_flooded_networks(rps,roads_TH,calc_path,output_path):

    # Read flooded road network for Thanh Hoa
    flooded_roads_path = os.path.join(output_path,'Thanh_Hoa_river_flooded_roads.shp')
    flooded_roads = gpd.read_file(flooded_roads_path)

    for rp in rps:
        if rp is not 'no_flood':
            # set path to save new shapefile
            flooded_road_path = os.path.join(calc_path,'Thanh_Hoa_river_flooded_roads_%s.shp' % rp)
            
            # only save if file does not exist yet
            if os.path.exists(flooded_road_path) is not True: 
                # get flooded roads
                rp_subset = flooded_roads.loc[flooded_roads[rp] > 0]
                
                # save road network without flooded roads
                roads_TH.loc[~roads_TH['edge_id'].isin(list(rp_subset['edge_id']))].to_file(flooded_road_path)


def line_length(line, ellipsoid='WGS-84'):
    """Length of a line in meters, given in geographic coordinates.

    Adapted from https://gis.stackexchange.com/questions/4022/looking-for-a-pythonic-way-to-calculate-the-length-of-a-wkt-linestring#answer-115285

    Args:
        line: a shapely LineString object with WGS-84 coordinates.
        
        ellipsoid: string name of an ellipsoid that `geopy` understands (see http://geopy.readthedocs.io/en/latest/#module-geopy.distance).

    Returns:
        Length of line in meters.
    """
    if line.geometryType() == 'MultiLineString':
        return sum(line_length(segment) for segment in line)

    return sum(
        vincenty(a, b, ellipsoid=ellipsoid).kilometers
        for a, b in pairwise(line.coords)
    )
    
 
def shortest_distance(rp,calc_path,road_path,communes_TH,places_TH,spatial_places):

    if rp is not 'no_flood':
        # read path to save new shapefile
        flooded_road_path = os.path.join(calc_path,'Thanh_Hoa_river_flooded_roads_%s.shp' % rp)
        # load network
        sg = nx.read_shp(flooded_road_path).to_undirected()
        
    else:
        g = nx.read_shp(road_path)
        sg = max(nx.connected_component_subgraphs(g.to_undirected()), key=len)

    commune_dist = {}
    for id_,commune in communes_TH.iterrows():
        
        # find nearest points of interest (places in this case)
        nearest_places = places_TH.iloc[list(spatial_places.nearest(commune.geometry.bounds,3))]
        
        k = 0
        for idplace,place in nearest_places.iterrows():
        
            # Compute the shortest path based on travel time
            try:
                path = nx.shortest_path(sg,
                                        source=tuple(commune['nearest_node']),
                                        target=tuple(place['nearest_node']),weight='t_time')
            except:
                path = []
            
            # save path
            if len(path) == 1:
                distance = 0
            elif len(path) == 0:
                distance = 9999
            else:
                distance = pd.DataFrame([sg[path[i]][path[i + 1]]
                                  for i in range(len(path) - 1)],
                                 columns=['length']).sum()[0]
                    
            if k == 0:
                shortest_distance = distance
            else:
                if distance < shortest_distance:
                    shortest_distance = distance
        
            k += 1
        
        commune_dist[commune['commune_id']] = shortest_distance

    # create dataframe of distances       
    commune_dist = pd.DataFrame.from_dict(commune_dist,orient='index')
    commune_dist.columns = [rp]
    
    return commune_dist

if __name__ == "__main__":
    main()
