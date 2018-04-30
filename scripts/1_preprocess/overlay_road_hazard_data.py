# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 08:52:05 2018

@author: elcok
"""

import os
from rasterstats import zonal_stats

import geopandas as gpd
import json

def main():
    # =============================================================================
    #     # Define current directory and data directory
    # =============================================================================
    config_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
    )
    with open(config_path, 'r') as config_fh:
        config = json.load(config_fh)
    data_path = config['paths']['data']
    output_path = config['paths']['output']

    # =============================================================================
    #     # Read road network for Thanh Hoa
    # =============================================================================
    road_path = os.path.join(data_path,'Roads','ThanhHao_roads','road_thanhhaoedges.shp')
    roads_TH = gpd.read_file(road_path)

    # =============================================================================
    #     # Read paths for river floods in Thanh Hoa
    # =============================================================================
    river_floods = [x for x in os.listdir(os.path.join(data_path,'Floods','Thanh_Hoa','Flood_Maps')) if x.endswith('.tiff')]

    # =============================================================================
    #     # Intersect roads with the floods
    # =============================================================================
    for flood in river_floods:
        flood_name = flood[:-5]
        flood_path = os.path.join(data_path,'Floods','Thanh_Hoa','Flood_Maps',flood)

        roads_TH[flood_name] = [x['mean'] for x in zonal_stats(roads_TH, flood_path,stats="mean",nodata=-9999)]

    flooded_roads = roads_TH.dropna(axis=0,how='all',subset =['10_freq','1_freq','20_freq','5_freq'])
    flooded_roads = flooded_roads.drop(['node_f_id', 'node_t_id','name', 'width', 'level','loads'],axis=1)
    
# =============================================================================
#     # Save the intersected roads
# =============================================================================
    flooded_roads.to_file(os.path.join(output_path,'Thanh_Hoa_river_flooded_roads.shp'))
   
if __name__ == '__main__':
    main()
    
    
