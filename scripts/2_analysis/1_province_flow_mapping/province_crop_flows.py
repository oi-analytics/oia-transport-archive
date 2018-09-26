# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:41:39 2018

@author: elcok
"""

import geopandas as gpd
import pandas as pd
import os
import igraph as ig
import numpy as np
import sys
import subprocess
from shapely.geometry import Point
from geoalchemy2 import Geometry, WKTElement



from vtra.utils import load_config,extract_value_from_gdf,get_nearest_node,gdf_clip,gdf_geom_clip,count_points_in_polygon
from vtra.transport_network_creation import province_shapefile_to_network, add_igraph_time_costs_province_roads

def cropflows_edges(region_name,start_points,end_points,graph,excel_sheet,excel_writer =''):
	"""
	Assign net revenue to roads assets in Vietnam

	Inputs are:
	start_points - GeoDataFrame of start points for shortest path analysis.
	end_points - GeoDataFrame of potential end points for shorest path analysis.
	G - iGraph network of the province.
	save_edges -

	Outputs are:
	Shapefile with all edges and the total net reveneu transferred along each edge
	GeoDataFrame of total net revenue transferred along each edge
	"""
	save_paths = []
	# path_index = 0
	for iter_,place in start_points.iterrows():
		try:
			closest_center = end_points.loc[end_points['OBJECTID']
			== place['NEAREST_C_CENTER']]['NEAREST_G_NODE'].values[0]

			pos0_i = graph.vs[node_dict[place['NEAREST_G_NODE']]]
			pos1_i = graph.vs[node_dict[closest_center]]

			if pos0_i != pos1_i:
				path = graph.get_shortest_paths(pos0_i,pos1_i,weights='min_cost',output="epath")
				get_od_pair = (place['NEAREST_G_NODE'],closest_center)
				get_path = [graph.es[n]['edge_id'] for n in path][0]
				get_dist = sum([graph.es[n]['length'] for n in path][0])
				get_time = sum([graph.es[n]['min_time'] for n in path][0])
				get_travel_cost = sum([graph.es[n]['min_cost'] for n in path][0])
				# path_index += 1
				save_paths.append((str(get_od_pair),str(get_path),get_travel_cost,get_dist,get_time,place['tons']))
		except:
			print(iter_)


	save_paths_df = pd.DataFrame(save_paths,columns = ['od_nodes','edge_path','travel_cost','distance','time','tons'])
	# print (save_paths_df)
	save_paths_df = save_paths_df.groupby(['od_nodes','edge_path','travel_cost','distance','time'])['tons'].sum().reset_index()
	save_paths_df.to_excel(excel_writer,excel_sheet,index = False)
	excel_writer.save()
	# del save_paths_df

	all_edges = [x['edge_id'] for x in graph.es]
	# all_edges_geom = [x['geometry'] for x in G.es]
	crop_tot = []
	for edge in all_edges:
		edge_path_index = list(set(save_paths_df.loc[save_paths_df['edge_path'].str.contains(edge)].index.tolist()))
		if edge_path_index:
			crop_tot.append(sum([save_paths_df.iloc[e]['tons'] for e in edge_path_index]))
		else:
			crop_tot.append(0)

	del save_paths_df

	gd_edges = pd.DataFrame(list(zip(all_edges,crop_tot)))
	gd_edges.columns = ['edge_id',excel_sheet]
	# gd_edges = pd.DataFrame(all_edges,all_edges_geom)
	# gd_edges.columns = ['edge_id']

	# gd_edges[crop_name] = 0
	# flow_dataframe.loc[flow_dataframe['edge_path'].str.contains(edge)].index.tolist()
	# for path in save_paths:
	# 	gd_edges.loc[gd_edges['edge_id'].isin(path[2]),crop_name] += path[3]

	# if save_edges == True:
	# 	gdf_edges.to_file(os.path.join(output_path,'weighted_edges_district_center_flows_{}.shp'.format(region_name)))
	return gd_edges



if __name__ == '__main__':

	data_path,calc_path,output_path = load_config()['paths']['data'],load_config()['paths']['calc'],load_config()['paths']['output']

	# provinces to consider
	province_list = ['Lao Cai','Binh Dinh','Thanh Hoa']
	# district_committe_names = ['district_people_committee_points_thanh_hoa.shp','district_province_peoples_committee_point_binh_dinh.shp','district_people_committee_points_lao_cai.shp']

	district_committe_names = ['district_people_committee_points_lao_cai.shp','district_province_peoples_committee_point_binh_dinh.shp','district_people_committee_points_thanh_hoa.shp']

	shp_output_path = os.path.join(output_path,'flow_mapping_shapefiles')

	province_path = os.path.join(data_path,'Vietnam_boundaries','who_boundaries','who_provinces.shp')
	provinces = gpd.read_file(province_path)
	provinces = provinces.to_crs({'init': 'epsg:4326'})

	crop_data_path = os.path.join(data_path,'Agriculture_crops','crop_data')
	crop_names = ['rice','cash','cass','teas','maiz','rubb','swpo','acof','rcof','pepp']

	for prn in range(len(province_list)):
		province = province_list[prn]
		# set all paths for all input files we are going to use
		province_name = province.replace(' ','').lower()

		edges_in = os.path.join(data_path,'Roads','{}_roads'.format(province_name),'vietbando_{}_edges.shp'.format(province_name))
		nodes_in = os.path.join(data_path,'Roads','{}_roads'.format(province_name),'vietbando_{}_nodes.shp'.format(province_name))
		commune_center_in = os.path.join(data_path,'Points_of_interest',district_committe_names[prn])

		flow_output_excel = os.path.join(output_path,'crop_flows','{}_province_roads_crop_flow_paths.xlsx'.format(province_name))
		excl_wrtr = pd.ExcelWriter(flow_output_excel)
		# commune_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','commune_level_stats.shp')

		# load provinces and get geometry of the right province
		province_geom = provinces.loc[provinces.NAME_ENG == province].geometry.values[0]

		prov_commune_center = gdf_clip(commune_center_in,province_geom)
		if 'OBJECTID' not in prov_commune_center.columns.values.tolist():
			prov_commune_center['OBJECTID'] = prov_commune_center.index

		print (prov_commune_center)

		# load nodes and edges
		nodes = gpd.read_file(nodes_in)
		nodes = nodes.to_crs({'init': 'epsg:4326'})
		sindex_nodes = nodes.sindex

		# Get nearest commune or district center
		prov_commune_center['NEAREST_G_NODE'] = prov_commune_center.geometry.apply(lambda x: get_nearest_node(x,sindex_nodes,nodes,'NODE_ID'))

		# load network

		# G = province_shapefile_to_network(edges_in,path_width_table)
		G = province_shapefile_to_network(edges_in)
		G = add_igraph_time_costs_province_roads(G,0.019)

		all_edges = [x['edge_id'] for x in G.es]
		all_edges_geom = [x['geometry'] for x in G.es]

		gdf_edges = pd.DataFrame(list(zip(all_edges,all_edges_geom)))
		gdf_edges.columns = ['edge_id','geometry']
		gdf_edges['crop_tot'] = 0

		for file in os.listdir(crop_data_path):
			if file.endswith(".tif") and 'spam_p' in file.lower().strip():
				fpath = os.path.join(crop_data_path, file)
				crop_name = [cr for cr in crop_names if cr in file.lower().strip()][0]
				raster_in = fpath
				outCSVName = os.path.join(output_path,'crop_flows','crop_concentrations.csv')
				# print ('gdal2xyz.py -csv '+raster_in+' '+ outCSVName)
				'''Clip to region and convert to points'''
				# os.system('gdal2xyz.py -csv '+raster_in+' '+ outCSVName)
				subprocess.run(["gdal2xyz.py",'-csv', raster_in,outCSVName])

				'''Load points and convert to geodataframe with coordinates'''
				load_points = pd.read_csv(outCSVName,header=None,names=['x','y','tons'],index_col=None)
				load_points = load_points[load_points['tons'] > 0]
				# load_points.to_csv('crop_concentrations.csv', index = False)

				geometry = [Point(xy) for xy in zip(load_points.x, load_points.y)]
				load_points = load_points.drop(['x', 'y'], axis=1)
				crs = {'init': 'epsg:4326'}
				crop_points = gpd.GeoDataFrame(load_points, crs=crs, geometry=geometry)
				# crop_points['geom'] = crop_points['geometry'].apply(lambda x: WKTElement(x.wkt, srid=4326))

				#drop the geometry column as it is now duplicative
				# crop_points.drop('geometry', 1, inplace=True)
				# points_gdp = points_gdp.rename(columns={'geometry':'geom'}).set_geometry('geom')
				del load_points
				# path_width_table = os.path.join(data_path,'Roads','road_properties','road_properties.xlsx')



				#clip all to province
				prov_crop = gdf_geom_clip(crop_points,province_geom)

				if len(prov_crop.index) > 0:
					# get revenue values for each village
					# prov_crop.to_file(os.path.join(output_path,'crop_flows','crop_tons.shp'))

					# first create sindex of all villages to count number of villages in commune
					prov_crop_sindex = prov_crop.sindex
					# and use average if commune has no stats
					# prov_pop.loc[prov_pop['netrev'] == 0,'netrev'] = prov_pop['netrev'].mean()

					# get nearest node in network for all start and end points
					prov_crop['NEAREST_G_NODE'] = prov_crop.geometry.apply(lambda x: get_nearest_node(x,sindex_nodes,nodes,'NODE_ID'))

					# prepare for shortest path routing, we'll use the spatial index of the centers
					# to find the nearest center for each population point
					# print (prov_crop)
					# print (prov_commune_center)
					sindex_commune_center = prov_commune_center.sindex
					prov_crop['NEAREST_C_CENTER'] = prov_crop.geometry.apply(lambda x: get_nearest_node(x,sindex_commune_center,prov_commune_center,'OBJECTID'))

					# get updated edges
					edges_updated = cropflows_edges(province_name,prov_crop,prov_commune_center,G,crop_name,excel_writer = excl_wrtr)
					gdf_edges = pd.merge(gdf_edges,edges_updated,how='left', on=['edge_id'])
					gdf_edges['crop_tot'] = gdf_edges['crop_tot'] + gdf_edges[crop_name]

					print ('Done with crop {0} in province {1}'.format(crop_name, province_name))

		gdf_edges = gpd.GeoDataFrame(gdf_edges,crs='epsg:4326')
		gdf_edges.to_file(os.path.join(shp_output_path,'weighted_edges_crop_flows_{}.shp'.format(province_name)))
