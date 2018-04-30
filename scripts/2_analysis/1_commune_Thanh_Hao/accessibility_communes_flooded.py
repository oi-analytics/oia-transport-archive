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

from functions import line_length

import cartopy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches


mpl.style.use('ggplot')
mpl.rcParams['font.size'] = 13.
mpl.rcParams['font.family'] = 'tahoma'
mpl.rcParams['axes.labelsize'] = 14.
mpl.rcParams['xtick.labelsize'] = 13.
mpl.rcParams['ytick.labelsize'] = 13.


if __name__ == "__main__":

# =============================================================================
#     # Define current directory and data directory
# =============================================================================
    base_path =   os.path.join(os.path.dirname(__file__),'..')    
    data_path = os.path.join(base_path,'Data')
    
# =============================================================================
#     # load commune geospatial dataset
# =============================================================================
    commune_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','commune_level_stats.shp')
    communes = gpd.read_file(commune_path)
    
# =============================================================================
#     #load provinces
# =============================================================================
    province_path = os.path.join(data_path,'Vietnam_boundaries','who_boundaries','who_provinces.shp')
    provinces = gpd.read_file(province_path)
    Thanh_Hoa = provinces.loc[provinces.NAME_ENG == 'Thanh Hoa']
    
# =============================================================================
#     # load communes Thanh Hoa and clean the dataframe a bit
# =============================================================================
    communes_TH = communes.loc[communes.pro_name_e == 'Thanh Hoa']
    communes_TH = communes_TH.drop(['gid', 'objectid','who_provin', 'prov_ec_id',
       'who_distri', 'dis_ec_id', 'name_vie', 'name_eng', 'dis_name_v',
       'dis_com_co', 'commune_ec', 'dis_name_e', 'pro_name_v','address', 'unnamed_ 1', 'unnamed__1','shape_leng', 'shape_le_1', 'shape_area'], axis=1)    

# =============================================================================
#     # Load places from OSM
# =============================================================================
    country_pbf = os.path.join(data_path,'OSM','vietnam-latest.osm.pbf')
    place_path = os.path.join(data_path,'OSM','places.shp')
    
    if os.path.isfile(place_path) is not True:
        os.system("ogr2ogr -progress -overwrite -f \"ESRI Shapefile\" -sql \
                   \"SELECT place FROM points WHERE place is not NULL\" \
                   -lco ENCODING=UTF-8 -skipfailures "+place_path+" "+country_pbf)    
    
    places = gpd.read_file(place_path)
    places.geometry = places.centroid
    
# =============================================================================
#     # clip places to Thanh Hoa (pretty efficient code if I may say so myself)
# =============================================================================
    places_TH = places.loc[places['geometry'].apply(lambda x: x.within(Thanh_Hoa['geometry'].values[0]))]
    
# =============================================================================
#     # only keep towns and bigger:
# =============================================================================
    places_TH = places_TH.loc[places_TH.place.isin(['town','city'])]
    
    places_TH.to_file(os.path.join(data_path,'OSM','places_TH.shp'))
    
    # create spatial index of towns and bigger file
    spatial_places = places_TH.sindex
    
# =============================================================================
#     # Read road network for Thanh Hoa
# =============================================================================
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

# =============================================================================
#     # look up nearest node for each centroid of commune and place
# =============================================================================
    def find_node(x):
        centroidal = list(x.centroid.coords)[:1][0]
        return nodes[np.argmin(np.sum((nodes - centroidal)**2, axis=1))]    
        
    # for communes
    communes_TH['nearest_node'] = communes_TH.geometry.apply(lambda x: find_node(x))
     
    #for places
    places_TH['nearest_node'] = places_TH.geometry.apply(lambda x: find_node(x))

# =============================================================================
#     # Read flooded road network for Thanh Hoa
# =============================================================================

    flooded_roads_path = os.path.join(base_path,'output','Thanh_Hoa_river_flooded_roads.shp')
    flooded_roads = gpd.read_file(flooded_roads_path)

    rps = ['no_flood','1_freq','5_freq','10_freq','20_freq']

# =============================================================================
#     # save flooded networks, saves computation time later
# =============================================================================
    for rp in rps:
        if rp is not 'no_flood':
            # set path to save new shapefile
            flooded_road_path = os.path.join(base_path,'calc','Thanh_Hoa_river_flooded_roads_%s.shp' % rp)
            
            # only save if file does not exist yet
            if os.path.exists(flooded_road_path) is not True: 
                # get flooded roads
                rp_subset = flooded_roads.loc[flooded_roads[rp] > 0]
                
                # save road network without flooded roads
                roads_TH.loc[~roads_TH['edge_id'].isin(list(rp_subset['edge_id']))].to_file(flooded_road_path)
 
# =============================================================================
#   # estimate distance to nearest place for each commune centroid
# =============================================================================
    distance_per_rp = {}
    for rp in rps:
        if rp is not 'no_flood':
            # read path to save new shapefile
            flooded_road_path = os.path.join(base_path,'calc','Thanh_Hoa_river_flooded_roads_%s.shp' % rp)
            # load network
            sg = nx.read_shp(flooded_road_path).to_undirected()
            
        else:
            g = nx.read_shp(road_path)
            sg = max(nx.connected_component_subgraphs(g.to_undirected()), key=len)

        nodes = np.array(sg.nodes())

        commune_dist = {}
        for id_,commune in communes_TH.iterrows():
            
             # find nearest points of interest (places in this case)
            nearest_places = places_TH.iloc[list(spatial_places.nearest(commune.geometry.bounds,3))]
            
            k = 0
            for idplace,place in nearest_places.iterrows():
      
                # define end point and the corresponding node in the network
                end_point = list(place.geometry.centroid.coords)[:1][0]
                pos1_i = np.argmin(np.sum((nodes - end_point)**2, axis=1))
            
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

    # =============================================================================
    #     # create dataframe of distances       
    # =============================================================================
        commune_dist = pd.DataFrame.from_dict(commune_dist,orient='index')
        commune_dist.columns = [rp]

        distance_per_rp[rp] = commune_dist
        
    # =============================================================================
    #     # merge with initial commune dataset
    # =============================================================================
    new_distances = pd.concat(distance_per_rp,axis=1)
    new_distances.columns = [x[0] for x in list(new_distances.columns)]
    communes_affected = communes_TH.merge(new_distances, left_on='commune_id',right_index=True)    
    communes_affected = communes_affected.drop('nearest_node',axis=1)

    # =============================================================================
    #     # create maps
    # =============================================================================
    for rp in rps:
        bins = [-1, 1, 5, 10, 25, 50,75,np.float('inf')]

        rp_names = {'no_flood': 'no flooding','1_freq': 'a 1/100 flood','5_freq' : 'a 1/20 flood',
                    '10_freq' : 'a 1/10 flood','20_freq': 'a 1/5 flood'}

        communes_affected['binned_{}'.format(rp)] = pd.cut(communes_affected[rp], bins=bins, labels=[0,1,2,3,4,5,6])
        communes_affected.loc[communes_affected.geometry.area.idxmin(),'binned_no_flood'] = 6

        # create figure        
        fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(8,6))

        # set projection
        proj_lat_lon = cartopy.crs.PlateCarree()
        ax = plt.axes([0.0,0.0,1.0, 1.0] ,facecolor='#D0E3F4', projection=proj_lat_lon)
    
        # set bounds and extent
        tot_bounds = list(communes_affected.total_bounds)
        ax.set_extent([tot_bounds[0]-0.1,tot_bounds[2]+0.1,tot_bounds[1]-0.1,tot_bounds[3]+0.1] , crs=proj_lat_lon)
        
        # load background 
        world = gpd.read_file(os.path.join('..','data','Vietnam_boundaries','who_boundaries','who_provinces.shp'))
    
        world.plot(ax=ax,color='#FEF9E0',lw=0.3,edgecolor='k')

        # create cmap
        cmap = cm.get_cmap('Reds', len(bins)) # Colour map (there are many others)
        cmaplist = ['#fee5d9','#fcbba1','#fc9272','#fb6a4a','#de2d26','#a50f15','#442c2d']
        cmap = cmap.from_list('Custom cmap', cmaplist, len(cmaplist))

        # plot figure
        communes_affected.plot(ax=ax,column='binned_{}'.format(rp),cmap=cmap)

        # create legend
        handles = []
        lnames = ['0-1 km','1-5 km','5-10 km','10-25 km','25-50 km','50-75 km', 'No Access'] #,'50-75 km',
        l = 0
        for color in cmaplist:
            handles.append(mpatches.Patch(color=color, label=lnames[l]))
            l += 1

        ax.legend(handles=handles,loc=3, prop={'size': 13}) 

        plt.title('Distance to major towns in Thanh Hoa for %s' % rp_names[rp],fontweight='bold')
    
        figure_out= os.path.join('..','Figures','Dist_Major_Towns_Thanh_Hoa_%s.png' % rp)
        plt.savefig(figure_out,dpi=600,bbox_inches='tight')

    # =============================================================================
    #     # create some summary statistics for affected industries and communes
    # =============================================================================

    sectors = ['nongnghiep', 'khaikhoang', 'chebien', 'detmay','gogiay', 'sanxuat', 'xaydung', 'thuongmai', 'dichvu']
    sectors_eng = ['agriculture', 'mining', 'processing', 'textile','wood & paper', 'manufacture', 'construction', 'trade', 'service']
    sectors_eng = [x.capitalize() for x in sectors_eng]
    
    for rp in rps:
        if rp == 'no_flood':
            continue
        only_communes_affected = communes_affected.loc[communes_affected['no_flood'] != communes_affected[rp]]
        
        only_communes_affected = only_communes_affected.copy()
        only_communes_affected[sectors] = only_communes_affected[sectors].multiply(only_communes_affected['nfirm'],axis='index')
        
        firms_affected = only_communes_affected[sectors].sum()

        firms_affected.index = sectors_eng

# =============================================================================
#         # create figure for total number of firms affected
# =============================================================================
        fig2,ax2 = plt.subplots(nrows=1, ncols=1,figsize=(6,6))

        cmaplist = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
        cmap = cmap.from_list('Custom cmap', cmaplist, len(cmaplist))

        firms_affected.plot(kind='pie',ax=ax2, fontsize=12,cmap=cmap,startangle=90, pctdistance=0.85)
        ax2.axis('equal')  

        plt.axis('off')

        plt.title('Relative distribution of industries affected in \n affected communes Thanh Hoa for %s' % rp_names[rp],fontweight='bold',fontsize=17)

        figure_out= os.path.join('..','Figures','Share_industries_affected_communes_Thanh_Hoa_%s.png' % rp)
        plt.savefig(figure_out,dpi=600,bbox_inches='tight')


# =============================================================================
#         # create figure for total number of employees affected
# =============================================================================
        only_communes_affected[sectors] = only_communes_affected[sectors].multiply(only_communes_affected['labor'],axis='index')
        
        firms_affected = only_communes_affected[sectors].sum()

        firms_affected.index = sectors_eng


        fig2,ax2 = plt.subplots(nrows=1, ncols=1,figsize=(6,6))

        cmaplist = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']
        cmap = cmap.from_list('Custom cmap', cmaplist, len(cmaplist))

        firms_affected.plot(kind='pie',ax=ax2, fontsize=12,cmap=cmap,startangle=90, pctdistance=0.85)
        ax2.axis('equal')  

        plt.axis('off')

        plt.title('Relative distribution of employees affected in \n affected communes Thanh Hoa for %s' % rp_names[rp],fontweight='bold',fontsize=17)

        figure_out= os.path.join('..','Figures','Share_employees_affected_communes_Thanh_Hoa_%s.png' % rp)
        plt.savefig(figure_out,dpi=600,bbox_inches='tight')
        

# =============================================================================
#         # create figure relative impact industries for Thanh Hao
# =============================================================================

        only_communes_affected = communes_affected.loc[communes_affected['no_flood'] != communes_affected[rp]]
        
        
        firms_affected = only_communes_affected[sectors].multiply(only_communes_affected['nfirm'],axis='index').sum()

        firms_affected.index = sectors_eng

        
        firms_affected_TH = communes_affected[sectors].multiply(communes_affected['nfirm'],axis='index').sum()
        firms_affected_TH.index = sectors_eng

        firms_affected_TH = (firms_affected/firms_affected_TH)*100

        fig3,ax3 = plt.subplots(nrows=1, ncols=1,figsize=(6,6))

        cmaplist = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']

        firms_affected_TH.plot(kind='bar',ax=ax3, fontsize=12,color=cmaplist)
        ax3.set_ylabel("Percentage affected", fontweight='bold',fontsize=15)
        ax3.set_xlabel("Industrial sector", fontweight='bold',fontsize=15)

        plt.title('Relative share of firms affected \n in Thanh Hoa for %s' % rp_names[rp],fontweight='bold',fontsize=17)

        figure_out= os.path.join('..','Figures','Share_firms_affected_Thanh_Hoa_%s.png' % rp)
        plt.savefig(figure_out,dpi=600,bbox_inches='tight')
       

        