# -*- coding: utf-8 -*-
"""
Python script to create transport networks in Vietnam
Created on Wed June 27 2018

@author: Raghav Pant, Elco Koks
"""

import pandas as pd
import os
import psycopg2
import networkx as nx
import csv
import igraph as ig
import numpy as np
import geopandas as gpd
from vtra.utils import line_length

def assign_province_road_conditions(x):
	asset_code = x.code
	asset_level = x.level

	if asset_code in (17,303) or asset_level in (0,1): # This is an expressway, national and provincial road
		return 'paved'
	else:			# Anything else not included above
		return 'unpaved'

def assign_assumed_width_to_province_roads_from_file(asset_width,width_range_list):
	'''
	Assign widths to roads assets in Vietnam
	The widths are assigned based on our understanding of:
	1. The reported width in the data which is not reliable
	2. A design specification based understanding of the assumed width based on ranges of values

	Inputs are:
	asset_width - Numeric value for width of asset
	width_range_list - List of tuples containing (from_width,to_width,assumed_width)

	Outputs are:
	assumed_width - assigned width of the raod asset based on design specifications
	'''

	assumed_width = asset_width
	for width_vals in width_range_list:
		if width_vals[0] <= assumed_width <= width_vals[1]:
			assumed_width = width_vals[2]
			break

	return assumed_width

def assign_assumed_width_to_province_roads(x):
	'''
	Assign widths to roads assets in Vietnam
	The widths are assigned based on our understanding of:
	1. The reported width in the data which is not reliable
	2. A design specification based understanding of the assumed width based on ranges of values

	Inputs are:
	asset_width - Numeric value for width of asset

	Outputs are:
	modified_width - assigned width of the road asset based on design specifications
	'''
	if 0 <= x.width < 4.25:
		return 3.5
	elif 4.25 <= x.width < 6.0:
		return 5.0
	elif 6.0 <= x.width < 8.0:
		return 7.0
	elif 8.0 <= x.width < 11.5:
		return 9.0
	elif 11.5 <= x.width < 17.5:
		return 14.0
	elif 17.5 <= x.width < 24.5:
		return 21.0
	elif 24.5 <= x.width < 100:
		return 9.0
	else:
		return x.width

def assign_asset_type_to_province_roads_from_file(asset_code,asset_type_list):
	'''
	Assign asset types to roads assets in Vietnam
	The types are assigned based on our understanding of:
	1. The reported asset code in the data

	Inputs are:
	asset code - Numeric value for code of asset

	Outputs are:
	asset type - Which is either of (Bridge,Dam,Culvert,Tunnel,Spillway,Road)
	'''
	asset_type = 'road'
	for asset in asset_type_list:
		if asset_code == asset[0]:
			asset_type = asset[2]
			break

	return asset_type


def assign_asset_type_to_province_roads(x):
	'''
	Assign asset types to roads assets in Vietnam
	The types are assigned based on our understanding of:
	1. The reported asset code in the data

	Inputs are:
	asset code - Numeric value for code of asset

	Outputs are:
	asset type - Which is either of (Bridge,Dam,Culvert,Tunnel,Spillway,Road)
	'''
	if x.code in (12,25):
		return 'Bridge'
	elif x.code == (23):
		return 'Dam'
	elif x.code == (24):
		return 'Culvert'
	elif x.code == (26):
		return 'Tunnel'
	elif x.code == (27):
		return 'Spillway'
	else:
		return 'Road'


def assign_minmax_travel_speeds_province_roads_apply(x):
	'''
	Assign travel speeds to roads assets in Vietnam
	The speeds are assigned based on our understanding of:
	1. The types of assets
	2. The levels of classification of assets: 0-National,1-Provinical,2-Local,3-Other
	3. The terrain where the assets are located: Flat or Mountain or No information

	Inputs are:
	asset_code - Numeric code for type of asset
	asset_level - Numeric code for level of asset
	asset_terrain - String value of the terrain of asset

	Outputs are:
	speed_min - Minimum assigned speed in km/hr
	speed_max - Maximum assigned speed in km/hr
	'''
	asset_code = x.code
	asset_level = x.level
	asset_terrain= x.terrain

	if (not asset_terrain) or ('flat' in  asset_terrain.lower()):
		if asset_code == 17: # This is an expressway
			return 100,120
		elif asset_code in (15,4): # This is a residential road or a mountain pass
			return 40,60
		elif asset_level == 0: # This is any other national network asset
			return 80,100
		elif asset_level == 1:# This is any other provincial network asset
			return 60,80
		elif asset_level == 2: # This is any other local network asset
			return 40,60
		else:			# Anything else not included above
			return 20,40

	else:
		if asset_level < 3:
			return 40, 60
		else:
			return 20,40

def assign_minmax_time_costs_province_roads_apply(x,cost_dataframe):
	'''
	'''
	asset_code = x.code
	asset_level = x.level
	asset_terrain= x.terrain

	min_time_cost = 0
	max_time_cost = 0
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		if cost_param.code == asset_code:
			min_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.max_speed)
			max_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.min_speed)
			break
		elif cost_param.level == asset_level and cost_param.terrain == asset_terrain:
			min_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.max_speed)
			max_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.min_speed)
			break

	return min_time_cost, max_time_cost


def assign_minmax_tariff_costs_province_roads_apply(x,cost_dataframe):
	'''
	Assign travel speeds to roads assets in Vietnam
	The speeds are assigned based on our understanding of:
	1. The types of assets
	2. The levels of classification of assets: 0-National,1-Provinical,2-Local,3-Other
	3. The terrain where the assets are located: Flat or Mountain or No information

	Inputs are:
	asset_code - Numeric code for type of asset
	asset_level - Numeric code for level of asset
	asset_terrain - String value of the terrain of asset

	Outputs are:
	speed_min - Minimum assigned speed in km/hr
	speed_max - Maximum assigned speed in km/hr
	tariff_min_usd	tariff_max_usd
	'''
	asset_code = x.code
	asset_level = x.level
	asset_terrain= x.terrain

	min_tariff_cost = 0
	max_tariff_cost = 0
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		if cost_param.code == asset_code:
			min_tariff_cost = 1.0*cost_param.tariff_min_usd*x.length
			max_tariff_cost = 1.0*cost_param.tariff_max_usd*x.length
			break
		elif cost_param.level == asset_level and cost_param.terrain == asset_terrain:
			min_tariff_cost = 1.0*cost_param.tariff_min_usd*x.length
			max_tariff_cost = 1.0*cost_param.tariff_max_usd*x.length
			break

	return min_tariff_cost, max_tariff_cost


def province_shapefile_to_dataframe(edges_in,road_terrain,road_properties_file):
	"""
	input parameters:
		edges_in : string of path to edges file/network file.

	output:
		SG: connected graph of the shapefile
	"""

	edges = gpd.read_file(edges_in)
	edges.columns = map(str.lower, edges.columns)

	# assgin asset terrain
	edges['terrain'] = road_terrain

	# assign road conditon
	edges['road_cond'] = edges.apply(assign_province_road_conditions,axis=1)

	# assign asset type
	asset_type_list = [tuple(x) for x in pd.read_excel(road_properties_file,sheet_name ='provincial').values]
	edges['asset_type'] = edges.code.apply(lambda x: assign_asset_type_to_province_roads_from_file(x,asset_type_list))

	# get the right linelength
	edges['length'] = edges.geometry.apply(line_length)

	# correct the widths of the road assets
	# get the width of edges
	width_range_list = [tuple(x) for x in pd.read_excel(road_properties_file,sheet_name ='widths').values]
	edges['width'] = edges.width.apply(lambda x: assign_assumed_width_to_province_roads_from_file(x,width_range_list))

	# assign minimum and maximum speed to network
	edges['speed'] = edges.apply(assign_minmax_travel_speeds_province_roads_apply,axis=1)
	edges[['min_speed', 'max_speed']] = edges['speed'].apply(pd.Series)
	edges.drop('speed',axis=1,inplace=True)

	# assign minimum and maximum travel time to network
	edges['min_time'] = edges['length']/edges['max_speed']
	edges['max_time'] = edges['length']/edges['min_speed']


	cost_values_df = pd.read_excel(road_properties_file,sheet_name ='costs')

	# assign minimum and maximum cost of time in USD to the network
	# the costs of time  = (unit cost of time in USD/hr)*(travel time in hr)
	edges['time_cost'] = edges.apply(lambda x: assign_minmax_time_costs_province_roads_apply(x,cost_values_df),axis = 1)
	edges[['min_time_cost', 'max_time_cost']] = edges['time_cost'].apply(pd.Series)
	edges.drop('time_cost',axis=1,inplace=True)

	# assign minimum and maximum cost of tonnage in USD/ton to the network
	# the costs of time  = (unit cost of tariff in USD/ton-km)*(length in km)
	edges['tariff_cost'] = edges.apply(lambda x: assign_minmax_tariff_costs_province_roads_apply(x,cost_values_df),axis = 1)
	edges[['min_tariff_cost', 'max_tariff_cost']] = edges['tariff_cost'].apply(pd.Series)
	edges.drop('tariff_cost',axis=1,inplace=True)

	# make sure that From and To node are the first two columns of the dataframe
	# to make sure the conversion from dataframe to igraph network goes smooth
	edges = edges.reindex(list(edges.columns)[2:]+list(edges.columns)[:2],axis=1)

	return edges


def province_shapefile_to_network(edges_in,road_terrain,road_properties_file):
	# create network from edge file
	edges = province_shapefile_to_dataframe(edges_in,road_terrain,road_properties_file)
	G = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:])

	# only keep connected network
	return G.clusters().giant()


def assign_national_road_terrain(x):
	terrain_type = x.dia_hinh__

	if terrain_type is None:
		return 'flat'
	elif 'flat' in terrain_type.lower().strip(): # Assume flat for all roads with no terrain
		return 'flat'
	else:			# Anything else not included above
		return 'mountain'

def assign_national_road_conditions(x):
	road_cond = x.loai_mat__

	if road_cond is None:
		return 'paved'
	elif 'asphalt' in road_cond.lower().strip(): # Assume asphalt for all roads with no condition
		return 'paved'
	else:			# Anything else not included above
		return 'unpaved'

def assign_national_road_class(x):
	road_class = x.capkth__ca
	vehicle_numbers = x.vehicle_co

	if road_class is None:
		if vehicle_numbers >= 6000:
			return 1
		elif 3000 <= vehicle_numbers < 6000:
			return 2
		elif 1000 <= vehicle_numbers < 3000:
			return 3
		elif 300 <= vehicle_numbers < 1000:
			return 4
		elif 50 <= vehicle_numbers < 300:
			return 5
		else:
			return 6
	else:
		if ',' in road_class:
			road_class = road_class.split(',')
		else:
			road_class = [road_class]

		class_1 = [rc for rc in road_class if rc == 'i']
		class_2 = [rc for rc in road_class if rc == 'ii']
		class_3 = [rc for rc in road_class if rc == 'iii']
		class_4 = [rc for rc in road_class if rc == 'iv']
		class_5 = [rc for rc in road_class if rc == 'v']
		class_6 = [rc for rc in road_class if rc == 'vi']

		if class_1:
			return 1
		elif class_2:
			return 2
		elif class_3:
			return 3
		elif class_4:
			return 4
		elif class_5:
			return 5
		elif class_6:
			return 6
		elif vehicle_numbers >= 6000:
			return 1
		elif 3000 <= vehicle_numbers < 6000:
			return 2
		elif 1000 <= vehicle_numbers < 3000:
			return 3
		elif 300 <= vehicle_numbers < 1000:
			return 4
		elif 50 <= vehicle_numbers < 300:
			return 5
		else:
			return 6


def assign_assumed_width_to_national_roads_from_file(x,flat_width_range_list,mountain_width_range_list):
	'''
	Assign widths to roads assets in Vietnam
	The widths are assigned based on our understanding of:
	1. The class of the road which is not reliable
	2. The number of lanes
	3. The terrain of the road

	Inputs are:
	x - dataframe row
	flat_width_range_list - List of tuples containing (from_width,to_width,assumed_width)

	Outputs are:
	assumed_width - assigned width of the raod asset based on design specifications
	'''

	road_class = x.road_class
	road_lanes = x.lanenum__s
	if road_lanes is None:
		road_lanes = 0
	else:
		road_lanes = int(road_lanes)

	road_terrain = x.terrain

	assumed_width = 3.5
	if road_terrain == 'flat':
		for vals in flat_width_range_list:
			if road_class == vals.road_class:
				if road_lanes > 0 and road_lanes <= 8:
					assumed_width = road_lanes*vals.lane_width + vals.median_strip + 2.0*vals.shoulder_width
				else:
					assumed_width = vals.road_width
				break

	else:
		for vals in mountain_width_range_list:
			if road_class == vals.road_class:
				if road_lanes > 0 and road_lanes <= 8:
					assumed_width = road_lanes*vals.lane_width + vals.median_strip + 2.0*vals.shoulder_width
				else:
					assumed_width = vals.road_width
				break

	return assumed_width

def assign_min_max_speeds_to_national_roads_from_file(x,flat_width_range_list,mountain_width_range_list):
	'''
	Assign speeds to national roads in Vietnam
	The speeds are assigned based on our understanding of:
	1. The class of the road
	2. The estimated speed from the CVTS data
	3. The terrain of the road

	Inputs are:
	x - dataframe row
	flat_width_range_list - List of tuples containing flat road properties
	mountain_width_range_list - List of tuples containing mountain road properties

	Outputs are:
	min and max speeds - assigned speeds of the road asset based on estimated speeds and design specifications
	'''

	road_class = x.road_class
	road_terrain = x.terrain
	est_speed = x.est_speed

	min_speed = est_speed
	max_speed = est_speed
	if road_terrain == 'flat':
		for vals in flat_width_range_list:
			if road_class == vals.road_class:
				if est_speed == 0:
					min_speed = vals.design_speed
					max_speed = vals.design_speed

				elif est_speed >= vals.design_speed:
					min_speed = vals.design_speed

				else:
					max_speed = vals.design_speed

				break

	else:
		for vals in mountain_width_range_list:
			if road_class == vals.road_class:
				if est_speed == 0:
					min_speed = vals.design_speed
					max_speed = vals.design_speed

				elif est_speed >= vals.design_speed:
					min_speed = vals.design_speed

				else:
					max_speed = vals.design_speed

				break

	return min_speed, max_speed

def assign_minmax_time_costs_national_roads_apply(x,cost_dataframe):
	'''
	'''
	if x.vehicle_co > 2000:
		asset_code = 17
	else:
		asset_code = 1

	asset_level = 1

	asset_terrain= x.terrain

	min_time_cost = 0
	max_time_cost = 0
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		if (cost_param.code == asset_code) and (cost_param.road_cond == x.road_cond):
			min_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.max_speed)
			max_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.min_speed)
			break
		elif (cost_param.level == asset_level) and (cost_param.terrain == asset_terrain) and (cost_param.road_cond == x.road_cond):
			min_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.max_speed)
			max_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.min_speed)
			break

	return min_time_cost, max_time_cost

def assign_minmax_tariff_costs_national_roads_apply(x,cost_dataframe):
	'''
	Assign travel speeds to roads assets in Vietnam
	The speeds are assigned based on our understanding of:
	1. The types of assets
	2. The levels of classification of assets: 0-National,1-Provinical,2-Local,3-Other
	3. The terrain where the assets are located: Flat or Mountain or No information

	Inputs are:
	asset_code - Numeric code for type of asset
	asset_level - Numeric code for level of asset
	asset_terrain - String value of the terrain of asset

	Outputs are:
	speed_min - Minimum assigned speed in km/hr
	speed_max - Maximum assigned speed in km/hr
	tariff_min_usd	tariff_max_usd
	'''
	# if x.vehicle_co > 2000:
	# 	asset_code = 17
	# else:
	# 	asset_code = 1

	# asset_level = 0
	# asset_terrain= x.terrain

	# min_tariff_cost = 0
	# max_tariff_cost = 0
	# cost_list = list(cost_dataframe.itertuples(index=False))
	# for cost_param in cost_list:
	# 	if (cost_param.code == asset_code) and (cost_param.road_cond == x.road_cond):
	# 		min_tariff_cost = 1.0*cost_param.tariff_min_usd*x.length
	# 		max_tariff_cost = 1.0*cost_param.tariff_max_usd*x.length
	# 		break
	# 	elif (cost_param.level == asset_level) and (cost_param.terrain == asset_terrain) and (cost_param.road_cond == x.road_cond):
	# 		min_tariff_cost = 1.0*cost_param.tariff_min_usd*x.length
	# 		max_tariff_cost = 1.0*cost_param.tariff_max_usd*x.length
	# 		break

	min_tariff_cost = 0
	max_tariff_cost = 0
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		if cost_param.vehicle_min <= x.vehicle_co < cost_param.vehicle_max:
			min_tariff_cost = 1.0*cost_param.tariff_min_usd*x.length
			max_tariff_cost = 1.0*cost_param.tariff_max_usd*x.length
			break

	return min_tariff_cost, max_tariff_cost

def national_road_shapefile_to_dataframe(edges_in,road_properties_file):
	"""
	input parameters:
		edges_in : string of path to edges file/network file.

	output:
		SG: connected graph of the shapefile
	"""

	edges = gpd.read_file(edges_in)
	edges.columns = map(str.lower, edges.columns)

	# assgin asset terrain
	edges['terrain'] = edges.apply(assign_national_road_terrain,axis=1)

	# assign road conditon
	edges['road_cond'] = edges.apply(assign_national_road_conditions,axis=1)

	# assign road class
	edges['road_class'] = edges.apply(assign_national_road_class,axis=1)

	# get the right linelength
	edges['length'] = edges.geometry.apply(line_length)

	# correct the widths of the road assets
	# get the width of edges
	flat_width_range_list = list(pd.read_excel(road_properties_file,sheet_name ='flat_terrain_designs').itertuples(index = False))
	mountain_width_range_list = list(pd.read_excel(road_properties_file,sheet_name ='mountain_terrain_designs').itertuples(index = False))

	edges['width'] = edges.apply(lambda x: assign_assumed_width_to_national_roads_from_file(x,flat_width_range_list,mountain_width_range_list),axis = 1)

	# assign minimum and maximum speed to network
	edges['speed'] = edges.apply(lambda x: assign_min_max_speeds_to_national_roads_from_file(x,flat_width_range_list,mountain_width_range_list),axis = 1)
	edges[['min_speed', 'max_speed']] = edges['speed'].apply(pd.Series)
	edges.drop('speed',axis=1,inplace=True)

	# assign minimum and maximum travel time to network
	edges['min_time'] = edges['length']/edges['max_speed']
	edges['max_time'] = edges['length']/edges['min_speed']


	cost_values_df = pd.read_excel(road_properties_file,sheet_name ='costs')

	# assign minimum and maximum cost of time in USD to the network
	# the costs of time  = (unit cost of time in USD/hr)*(travel time in hr)
	edges['time_cost'] = edges.apply(lambda x: assign_minmax_time_costs_national_roads_apply(x,cost_values_df),axis = 1)
	edges[['min_time_cost', 'max_time_cost']] = edges['time_cost'].apply(pd.Series)
	edges.drop('time_cost',axis=1,inplace=True)

	# assign minimum and maximum cost of tonnage in USD/ton to the network
	# the costs of time  = (unit cost of tariff in USD/ton-km)*(length in km)
	edges['tariff_cost'] = edges.apply(lambda x: assign_minmax_tariff_costs_national_roads_apply(x,cost_values_df),axis = 1)
	edges[['min_tariff_cost', 'max_tariff_cost']] = edges['tariff_cost'].apply(pd.Series)
	edges.drop('tariff_cost',axis=1,inplace=True)

	# make sure that From and To node are the first two columns of the dataframe
	# to make sure the conversion from dataframe to igraph network goes smooth
	edges = edges.reindex(list(edges.columns)[2:]+list(edges.columns)[:2],axis=1)

	return edges

def national_shapefile_to_dataframe():
	# called from analysis.national_flow_mapping.national_industry_flows
	raise NotImplementedError()

def national_road_shapefile_to_network(edges_in,road_properties_file):
	# create network from edge file
	edges = national_road_shapefile_to_dataframe(edges_in,road_properties_file)
	G = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:])

	# only keep connected network
	return G.clusters().giant()

def add_igraph_generalised_costs_roads(G,vehicle_numbers,tonnage):
	# G.es['max_cost'] = list(cost_param*(np.array(G.es['length'])/np.array(G.es['max_speed'])))
	# G.es['min_cost'] = list(cost_param*(np.array(G.es['length'])/np.array(G.es['min_speed'])))
	# print (G.es['max_time'])
	G.es['max_gcost'] = list(vehicle_numbers*(np.array(G.es['max_time_cost'])) + tonnage*(np.array(G.es['max_tariff_cost'])))
	G.es['min_gcost'] = list(vehicle_numbers*(np.array(G.es['min_time_cost'])) + tonnage*(np.array(G.es['min_tariff_cost'])))

	return G

def add_igraph_generalised_costs_province_roads():
	# called from failure.{adapt_results_process,adaptation_options_multi,failure_estimation_provinces_multi,failure_projections-multi,failure_scenario_generation}
	raise NotImplementedError()

def add_igraph_time_costs_province_roads():
	# called from analysis.province_flow_mapping.{commune_poi_analysis,province_crop_flows}
	raise NotImplementedError()

def add_igraph_generalised_costs_network(G,vehicle_numbers,tonnage,operating_factor_min,operating_factor_max):
	# G.es['max_cost'] = list(cost_param*(np.array(G.es['length'])/np.array(G.es['max_speed'])))
	# G.es['min_cost'] = list(cost_param*(np.array(G.es['length'])/np.array(G.es['min_speed'])))
	# print (G.es['max_time'])
	G.es['max_gcost'] = list((1 + operating_factor_max)*(vehicle_numbers*(np.array(G.es['max_time_cost'])) + tonnage*(np.array(G.es['max_tariff_cost']))))
	G.es['min_gcost'] = list((1 + operating_factor_min)*(vehicle_numbers*(np.array(G.es['min_time_cost'])) + tonnage*(np.array(G.es['min_tariff_cost']))))

	return G

def add_generalised_costs_network_dataframe(network_dataframe,vehicle_numbers,tonnage,operating_factor_min,operating_factor_max):
	# G.es['max_cost'] = list(cost_param*(np.array(G.es['length'])/np.array(G.es['max_speed'])))
	# G.es['min_cost'] = list(cost_param*(np.array(G.es['length'])/np.array(G.es['min_speed'])))
	# print (G.es['max_time'])
	network_dataframe['max_gcost'] = (1 + operating_factor_max)*(vehicle_numbers*network_dataframe['max_time_cost'] + tonnage*network_dataframe['max_tariff_cost'])
	network_dataframe['min_gcost'] = (1 + operating_factor_max)*(vehicle_numbers*network_dataframe['min_time_cost'] + tonnage*network_dataframe['min_tariff_cost'])

	return network_dataframe

def assign_minmax_time_costs_networks_apply(x,cost_dataframe):
	'''
	'''
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		min_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.max_speed)
		max_time_cost = 1.0*cost_param.time_cost_usd*(x.length/x.min_speed)


	return min_time_cost, max_time_cost

def assign_minmax_tariff_costs_networks_apply(x,cost_dataframe):
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		min_tariff_cost = 1.0*cost_param.tariff_min_usd*x.length
		max_tariff_cost = 1.0*cost_param.tariff_max_usd*x.length

	return min_tariff_cost, max_tariff_cost

def network_shapefile_to_dataframe(edges_in,mode_properties_file,mode_name,speed_min,speed_max):
	"""
	input parameters:
		edges_in : string of path to edges file/network file.

	output:
		SG: connected graph of the shapefile
	"""

	edges = gpd.read_file(edges_in)
	edges.columns = map(str.lower, edges.columns)

	# assgin asset terrain

	# get the right linelength
	edges['length'] = edges.geometry.apply(line_length)

	# assign some speeds
	edges['min_speed'] = speed_min
	edges['max_speed'] = speed_max

	# assign minimum and maximum travel time to network
	edges['min_time'] = edges['length']/edges['max_speed']
	edges['max_time'] = edges['length']/edges['min_speed']


	cost_values_df = pd.read_excel(mode_properties_file,sheet_name=mode_name)

	# assign minimum and maximum cost of time in USD to the network
	# the costs of time  = (unit cost of time in USD/hr)*(travel time in hr)
	edges['time_cost'] = edges.apply(lambda x: assign_minmax_time_costs_networks_apply(x,cost_values_df),axis = 1)
	edges[['min_time_cost', 'max_time_cost']] = edges['time_cost'].apply(pd.Series)
	edges.drop('time_cost',axis=1,inplace=True)

	# assign minimum and maximum cost of tonnage in USD/ton to the network
	# the costs of time  = (unit cost of tariff in USD/ton-km)*(length in km)
	edges['tariff_cost'] = edges.apply(lambda x: assign_minmax_tariff_costs_networks_apply(x,cost_values_df),axis = 1)
	edges[['min_tariff_cost', 'max_tariff_cost']] = edges['tariff_cost'].apply(pd.Series)
	edges.drop('tariff_cost',axis=1,inplace=True)

	# make sure that From and To node are the first two columns of the dataframe
	# to make sure the conversion from dataframe to igraph network goes smooth
	edges = edges.reindex(list(edges.columns)[2:]+list(edges.columns)[:2],axis=1)

	return edges

def network_shapefile_to_network(edges_in,mode_properties_file,mode_name,speed_min,speed_max):
	# create network from edge file
	edges = network_shapefile_to_dataframe(edges_in,mode_properties_file,mode_name,speed_min,speed_max)
	G = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:])

	# only keep connected network
	return G.clusters().giant()

def assign_minmax_tariff_costs_multi_modal_apply(x,cost_dataframe):
	min_tariff_cost = 0
	max_tariff_cost = 0
	cost_list = list(cost_dataframe.itertuples(index=False))
	for cost_param in cost_list:
		if cost_param.one_mode == x.port_type and cost_param.other_mode == x.to_mode:
			min_tariff_cost = cost_param.tariff_min_usd
			max_tariff_cost = cost_param.tariff_max_usd
			break
		elif cost_param.one_mode == x.to_mode and cost_param.other_mode == x.from_mode:
			min_tariff_cost = cost_param.tariff_min_usd
			max_tariff_cost = cost_param.tariff_max_usd
			break
		elif cost_param.one_mode == x.to_mode and cost_param.other_mode == x.port_type:
			min_tariff_cost = cost_param.tariff_min_usd
			max_tariff_cost = cost_param.tariff_max_usd
			break
		elif cost_param.one_mode == x.from_mode and cost_param.other_mode == x.to_mode:
			min_tariff_cost = cost_param.tariff_min_usd
			max_tariff_cost = cost_param.tariff_max_usd
			break

	return min_tariff_cost, max_tariff_cost

def multi_modal_shapefile_to_dataframe(edges_in,mode_properties_file,mode_name,length_threshold):
	"""
	input parameters:
		edges_in : string of path to edges file/network file.

	output:
		SG: connected graph of the shapefile
	"""

	edges = gpd.read_file(edges_in)
	edges.columns = map(str.lower, edges.columns)

	# assgin asset terrain

	# get the right linelength
	edges['length'] = edges.geometry.apply(line_length)


	cost_values_df = pd.read_excel(mode_properties_file,sheet_name=mode_name)

	# assign minimum and maximum cost of tonnage in USD/ton to the network
	# the costs of time  = (unit cost of tariff in USD/ton)
	edges['tariff_cost'] = edges.apply(lambda x: assign_minmax_tariff_costs_multi_modal_apply(x,cost_values_df),axis = 1)
	edges[['min_tariff_cost', 'max_tariff_cost']] = edges['tariff_cost'].apply(pd.Series)
	edges.drop('tariff_cost',axis=1,inplace=True)

	edges['min_time'] = 0
	edges['max_time'] = 0
	edges['min_time_cost'] = 0
	edges['max_time_cost'] = 0

	# make sure that From and To node are the first two columns of the dataframe
	# to make sure the conversion from dataframe to igraph network goes smooth
	edges = edges.reindex(list(edges.columns)[2:]+list(edges.columns)[:2],axis=1)
	edges = edges[edges['length'] < length_threshold]

	return edges

def multi_modal_shapefile_to_network(edges_in,mode_properties_file,mode_name,length_threshold):
	# create network from edge file
	edges = multi_modal_shapefile_to_dataframe(edges_in,mode_properties_file,mode_name,length_threshold)
	G = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:])

	# only keep connected network
	return G

'''
Functions we are not using at present for provincial analysis. Will clean them later
'''

def assign_minmax_travel_speeds_province_roads(asset_code,asset_level,asset_terrain):
	'''
	Assign travel speeds to roads assets in Vietnam
	The speeds are assigned based on our understanding of:
	1. The types of assets
	2. The levels of classification of assets: 0-National,1-Provincial,2-Local,3-Other
	3. The terrain where the assets are located: Flat or Mountain or No information

	Inputs are:
	asset_code - Numeric code for type of asset
	asset_level - Numeric code for level of asset
	asset_terrain - String value of the terrain of asset

	Outputs are:
	speed_min - Minimum assigned speed in km/hr
	speed_max - Maximum assigned speed in km/hr
	'''

	if (not asset_terrain) or (asset_terrain == 'flat'):
		if asset_code == 17: # This is an expressway
			return 100,120
		elif asset_code in (15,4): # This is a residential road or a mountain pass
			return 40,60
		elif asset_level == 0: # This is any other national network asset
			return 80,100
		elif asset_level == 1:# This is any other provincial network asset
			return 60,80
		elif asset_level == 2: # This is any other local network asset
			return 40,60
		else:			# Anything else not included above
			return 20,40

	else:
		if asset_level < 3:
			return 40, 60
		else:
			return 20,40

def shapefile_to_network(edges_in,path_width_table):
	"""
	input parameters:
		edges_in : string of path to edges file/network file.

	output:
		SG: connected graph of the shapefile
	"""

	edges = gpd.read_file(edges_in)

	# assign minimum and maximum speed to network
	edges['SPEED'] = edges.apply(assign_minmax_travel_speeds_roads_apply,axis=1)
	edges[['MIN_SPEED', 'MAX_SPEED']] = edges['SPEED'].apply(pd.Series)
	edges.drop('SPEED',axis=1,inplace=True)

	# get the right linelength
	edges['LENGTH'] = edges.geometry.apply(line_length)

	# get the width of edges
	width_range_list = [tuple(x) for x in pd.read_excel(path_width_table,sheet_name ='widths').values]

	edges['width'] = edges.width.apply(lambda x: assign_assumed_width_to_roads(x,width_range_list))

	# make sure that From and To node are the first two columns of the dataframe
	# to make sure the conversion from dataframe to igraph network goes smooth
	edges = edges.reindex(list(edges.columns)[2:]+list(edges.columns)[:2],axis=1)

	# create network from edge file
	G = ig.Graph.TupleList(edges.itertuples(index=False), edge_attrs=list(edges.columns)[2:])

	# only keep connected network
	return G.clusters().giant()

def assign_travel_times_and_variability(speed_attributes,variability_attributes,mode_type,variability_location,mode_attribute,distance):
	travel_time = 0
	variability_time = 0

	st = [s[2] for s in speed_attributes if s[0].lower().strip() == mode_type and s[1].lower().strip() == mode_attribute]
	if len(st) > 0:
		travel_time = distance/sum(st)
		vt = [v[3] for v in variability_attributes if v[0] == mode_type and v[1] == variability_location]
		# print (vt)
		variability_time = (1.0*sum(vt)/100)*travel_time

	return travel_time, variability_time

def assign_network_dictionary(network_dictionary,edge,from_node,to_node,distance,speed):

	network_dictionary['edge'].append(edge)
	network_dictionary['from_node'].append(from_node)
	network_dictionary['to_node'].append(to_node)
	network_dictionary['distance'].append(distance)
	network_dictionary['speed'].append(speed)
	network_dictionary['travel_cost'].append(0.019*distance/speed)

	return network_dictionary

def create_network_dictionary(network_dictionary,layer_name, edge_id, from_node_id, to_node_id,edge_speed_attribute,edge_geom_attribute,cursor,connection):
	'''
	input parameters:
	layer_name: sql layer name to extraxt information from
	edge_id: unique ID of edge in sql layer
	from_node_id: unqiue source node ID corresponding to edge ID in sql layer
	to_node_id: unique target node ID corresponding to edge ID in sql layer
	edge_geom_attribute: for calculating the length of the edge
	cursor: the postGIS cursor
	connection: the postGIS connection

	output:
	network_dict = {'edgeid':[],'from_node':[],'to_node':[],'distance':[]}

	'''
	sql_query = '''select {0}, {1}, {2}, {3}, st_length({4}::geography)/1000 from {5}
				'''.format(edge_id,from_node_id,to_node_id,edge_speed_attribute,edge_geom_attribute,layer_name)

	cursor.execute(sql_query)
	read_layer = cursor.fetchall()
	for row in read_layer:
		if str(row[0]).isdigit():
			e_id = int(row[0])
		else:
			e_id = row[0]

		if str(row[1]).isdigit():
			fn_id = int(row[1])
		else:
			fn_id = row[1]

		if str(row[2]).isdigit():
			tn_id = int(row[2])
		else:
			tn_id = row[2]

		sp = float(row[3])
		lgth = float(row[4])


		network_dictionary = assign_network_dictionary(network_dictionary,e_id,fn_id,tn_id,lgth,sp)

	return network_dictionary

def create_networkx_topology(network_dictionary,tonnage,teus):
	# all_net_dict = {'edge':[],'from_node':[],'to_node':[],'waiting_cost':[],'travel_cost':[],'transport_price_ton':[],
	# 			'transport_price_teu':[],'variability_cost':[]}

	G = nx.Graph()
	edges = network_dictionary['edge']
	from_nodes = network_dictionary['from_node']
	to_nodes = network_dictionary['to_node']
	wait_costs = network_dictionary['waiting_cost']
	travel_costs = network_dictionary['travel_cost']
	price_tons = network_dictionary['transport_price_ton']
	price_teus = network_dictionary['transport_price_teu']
	variable_costs = network_dictionary['variability_cost']

	for e in range(len(edges)):
		generalised_cost = wait_costs[e] + travel_costs[e] + tonnage*price_tons[e] + teus*price_teus[e] + variable_costs[e]
		G.add_edge(from_nodes[e],to_nodes[e], edge = edges[e], cost = generalised_cost)

	return G

def create_igraph_topology(network_dictionary):
	# all_net_dict = {'edge':[],'from_node':[],'to_node':[],'waiting_cost':[],'travel_cost':[],'transport_price_ton':[],
	# 			'transport_price_teu':[],'variability_cost':[]}

	G = ig.Graph()
	edges = network_dictionary['edge']
	from_nodes = network_dictionary['from_node']
	to_nodes = network_dictionary['to_node']

	unique_nodes = list(set(from_nodes + to_nodes))
	igraph_ids = list(range(len(unique_nodes)))
	G.add_vertices(igraph_ids)

	edge_list = []
	for e in range(len(edges)):
		fn = from_nodes[e]
		tn = to_nodes[e]
		ig_fn = igraph_ids[unique_nodes.index(fn)]
		ig_tn = igraph_ids[unique_nodes.index(tn)]

		edge_list.append((ig_fn,ig_tn))

	G.add_edges(edge_list)

	G.vs['node'] = unique_nodes

	G.es['from_node'] = from_nodes
	G.es['to_node'] = to_nodes
	G.es['edge'] = edges
	G.es['distance'] = network_dictionary['distance']
	G.es['travel_cost'] = network_dictionary['travel_cost']

	return G

def add_igraph_costs(G,tonnage,teus):
	# all_net_dict = {'edge':[],'from_node':[],'to_node':[],'waiting_cost':[],'travel_cost':[],'transport_price_ton':[],
	# 			'transport_price_teu':[],'variability_cost':[]}

	if teus > 0:
		generalised_cost = np.array(G.es['waiting_cost']) + np.array(G.es['travel_cost']) + teus*np.array(G.es['price_teus']) + np.array(G.es['variable_costs'])
	else:
		generalised_cost = np.array(G.es['waiting_cost'])+ np.array(G.es['travel_cost']) + tonnage*np.array(G.es['price_tons']) + np.array(G.es['variable_costs'])


	G.es['cost'] = list(generalised_cost)

	return G

def get_networkx_edges(G, node_path,data_cond):
	node_tup = zip(node_path[:-1],node_path[1:])
	t_list = []
	for tup in node_tup:
		t = [d for (u,v,d) in G.edges(data = data_cond) if (u,v) == tup or (v,u) == tup]
		if len(t) > 0:
			t_list.append(t[0])
	return t_list

def get_igraph_edges(G,edge_path,data_cond):
	t_list = [G.es[n][data_cond] for n in edge_path]
	return (t_list)
