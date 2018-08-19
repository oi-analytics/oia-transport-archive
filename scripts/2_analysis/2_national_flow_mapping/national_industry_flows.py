"""Summarise hazard data

Get OD data and process it
Author: Raghav Pant
Date: April 20, 2018
"""
import geopandas as gpd
import pandas as pd
import os
import igraph as ig
import numpy as np
import sys
import subprocess
from shapely.geometry import Point
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
import itertools
import operator
import ast
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from scripts.utils import load_config,extract_value_from_gdf,get_nearest_node,gdf_clip,gdf_geom_clip,count_points_in_polygon,voronoi_finite_polygons_2d,extract_nodes_within_gdf,assign_value_in_area_proportions
from scripts.transport_network_creation import province_shapefile_to_network, add_igraph_generalised_costs_province_roads


'''
Create the database connection
'''
def main():
	data_path,calc_path,output_path = load_config()['paths']['data'],load_config()['paths']['calc'],load_config()['paths']['output']

	exchange_rate = 1.05*(1000000/21000)

	# population_points_in = os.path.join(data_path,'Points_of_interest','population_points.shp')
	# commune_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','commune_level_stats.shp')

	# crop_data_path = os.path.join(data_path,'Agriculture_crops','crop_data')
	# rice_month_file = os.path.join(data_path,'rice_atlas_vietnam','rice_production.shp')
	# crop_month_fields = ['P_Jan','P_Feb','P_Mar','P_Apr','P_May','P_Jun','P_Jul','P_Aug','P_Sep','P_Oct','P_Nov','P_Dec']
	# crop_names = ['rice','cash','cass','teas','maiz','rubb','swpo','acof','rcof','pepp']

	'''
	Get the modal shares
	'''
	od_data_modes = pd.read_excel(od_data_file,sheet_name = 'mode').fillna(0)
	# od_data_modes.columns = map(str.lower, od_data_modes.columns)
	o_id_col = 'o'
	d_id_col = 'd'
	od_data_modes['total'] = od_data_modes[mode_cols].sum(axis=1)
	for m in mode_cols:
		od_data_modes[m] = od_data_modes[m]/od_data_modes['total'].replace(np.inf, 0)

	od_data_modes['water'] = od_data_modes['inland'] + od_data_modes['coastal']		
	od_data_modes = od_data_modes.fillna(0)
	# od_data_modes.to_csv('mode_frac.csv',index = False)

	od_fracs = od_data_modes[new_mode_cols]

	od_data_com = pd.read_excel(od_data_file,sheet_name = 'goods').fillna(0)
	ind_cols = ['sugar','wood','steel','constructi','cement','fertilizer','coal','petroluem','manufactur','fishery','meat']
	od_fracs = pd.merge(od_fracs,od_data_com,how='left', on=['o','d']).fillna(0)

	del od_data_com,od_data_modes

	province_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','province_level_stats.shp')
	commune_path = os.path.join(data_path,'Vietnam_boundaries','boundaries_stats','commune_level_stats.shp')

	# load provinces and get geometry of the right province
	provinces = gpd.read_file(province_path)
	provinces = provinces.to_crs({'init': 'epsg:4326'})
	sindex_provinces = provinces.sindex
	
	# load provinces and get geometry of the right province
	communes = gpd.read_file(commune_path)
	communes = communes.to_crs({'init': 'epsg:4326'})
	sindex_communes = communes.sindex

	# modes_file_paths = [('Roads','national_roads'),('Railways','national_rail'),('Airports','airnetwork'),('Waterways','waterways')]
	modes_file_paths = [('Roads','national_roads')]
	for m in range(len(modes_file_paths)):
		mode_data_path = os.path.join(data_path,modes_file_paths[m][0],modes_file_paths[m][1])
		for file in os.listdir(mode_data_path):
			try:
				if file.endswith(".shp") and 'edges' in file.lower().strip():
					edges_in = os.path.join(mode_data_path, file)
				if file.endswith(".shp") and 'nodes' in file.lower().strip():
					nodes_in = os.path.join(mode_data_path, file)
			except:
				print ('Network nodes and edge files necessary')
			
		
		# load nodes of the network
		nodes = gpd.read_file(nodes_in)
		nodes = nodes.to_crs({'init': 'epsg:4326'})
		nodes.columns = map(str.lower, nodes.columns) 
		sindex_nodes = nodes.sindex

		# assign province ID's and OD ID's to their nearest nodes
		nodes['provinceid'] = nodes.geometry.apply(lambda x: get_nearest_node(x,sindex_provinces,provinces,'provinceid'))
		nodes['od_id'] = nodes.geometry.apply(lambda x: get_nearest_node(x,sindex_provinces,provinces,'od_id'))

		xy_list = []
		for iter_,values in nodes.iterrows():
			# print (list(values.geometry.coords))
			xy = list(values.geometry.coords)
			xy_list += [list(xy[0])]

		vor = Voronoi(np.array(xy_list))
		regions, vertices = voronoi_finite_polygons_2d(vor)
		min_x = vor.min_bound[0] - 0.1
		max_x = vor.max_bound[0] + 0.1
		min_y = vor.min_bound[1] - 0.1
		max_y = vor.max_bound[1] + 0.1

		mins = np.tile((min_x, min_y), (vertices.shape[0], 1))
		bounded_vertices = np.max((vertices, mins), axis=0)
		maxs = np.tile((max_x, max_y), (vertices.shape[0], 1))
		bounded_vertices = np.min((bounded_vertices, maxs), axis=0)

		box = Polygon([[min_x, min_y], [min_x, max_y], [max_x, max_y], [max_x, min_y]])
		# colorize
		poly_list = []
		for region in regions:
			polygon = vertices[region]
			# Clipping polygon
			poly = Polygon(polygon)
			poly = poly.intersection(box)
			poly_list.append(poly)
	

		poly_index = list(np.arange(0,len(poly_list),1))
		poly_df = pd.DataFrame(list(zip(poly_index,poly_list)),columns = ['gid','geometry'])
		gdf_voronoi = gpd.GeoDataFrame(poly_df,crs='epsg:4326')
		gdf_voronoi['node_id'] = gdf_voronoi.apply(lambda x: extract_nodes_within_gdf(x,nodes,'node_id'),axis = 1)
		gdf_voronoi['population'] = 0
		gdf_voronoi = assign_value_in_area_proportions(communes,gdf_voronoi,'population')

		gdf_pops = gdf_voronoi[['node_id','population']]
		nodes = pd.merge(nodes,gdf_pops,how='left', on=['node_id']).fillna(0)

		del gdf_pops

		nodes = nodes[['node_id','od_id','population']]
		nodes_sums = nodes.groupby(['od_id','node_id']).agg({'population': 'sum'})
		nodes_frac = nodes_sums.groupby(level = 0).apply(lambda x: x/float(x.sum()))
		nodes_frac = nodes_frac.reset_index(level = ['od_id','node_id'])
		nodes_frac.rename(columns={'population': 'pop_frac'}, inplace=True)

		nodes = pd.merge(nodes,nodes_frac[['node_id','pop_frac']],how='left', on=['node_id']).fillna(0)

		del nodes_frac

		for ind in ind_cols:
			ind_mode = modes[m]+ '_' + ind
			od_fracs[ind_mode] = od_fracs[modes[m]]*od_fracs[ind]
			

			od_flows = list(zip(od_fracs[o_id_col].values.tolist(),od_fracs[d_id_col].values.tolist(),od_fracs[ind_mode].values.tolist()))
			origins = list(set(od_fracs[o_id_col].values.tolist()))
			destinations = list(set(od_fracs[d_id_col].values.tolist()))

			dflows = []
			# print (od_flows)
			for o in origins:
				for d in destinations:
					fval = [fl for (org,des,fl) in od_flows if org == o and des == d]
					if len(fval) == 1 and fval[0] > 0:
						o_matches = [(item[0],item[2]) for item in od_nodes_regions if item[1] == o]
						if len(o_matches) > 0:
							for o_vals in o_matches:
								o_val = 1.0*fval[0]*(1.0*o_vals[1]/100)
								o_node = o_vals[0]
								d_matches = [(item[0],item[2]) for item in od_nodes_regions if item[1] == d]
								if len(d_matches) > 0:
									for d_vals in d_matches:
										od_val = 1.0*o_val*(1.0*d_vals[1]/100)
										d_node = d_vals[0]
										if od_val > 0 and o_node != d_node:
											# od_net = tnc.add_igraph_costs(od_net,t_val,0)
											orgn_node = od_net.vs['node'].index(o_node)
											dest_node = od_net.vs['node'].index(d_node)

											# n_pth = od_net.get_shortest_paths(orgn_node,to = dest_node, weights = 'travel_cost', mode = 'OUT', output='vpath')[0]
											e_pth = od_net.get_shortest_paths(orgn_node,to = dest_node, weights = 'travel_cost', mode = 'OUT', output='epath')[0]

											# n_list = [od_net.vs[n]['node'] for n in n_pth]
											e_list = [od_net.es[n]['edge'] for n in e_pth]
											# cst = sum([od_net.es[n]['cost'] for n in e_pth])
											net_dict = {'Origin_id':o_node,'Destination_id':d_node,'Origin_region':o,'Destination_region':d,
														'Tonnage':od_val,'edge_path':e_list,'node_path':[o_node,d_node]}
											wr.writerow(net_dict.values())
											dflows.append((str([o_node,d_node]),str(e_list),od_val))
									

					print (o,d,fval,modes[m],ind)


			node_table = modes[m] + '_node_flows'
			edge_table = modes[m] + '_edge_flows'

			# dom_flows = pd.read_csv(ofile).fillna(0)
			dom_flows = pd.DataFrame(dflows,columns = ['node_path', 'edge_path','Tonnage'])

			flow_node_edge = dom_flows.groupby(['node_path', 'edge_path'])['Tonnage'].sum().reset_index()
			n_dict = {}
			e_dict = {}
			n_dict,e_dict = get_node_edge_flows(flow_node_edge,n_dict,e_dict)

			node_list = get_id_flows(n_dict)
			df = pd.DataFrame(node_list, columns = ['node_id',ind])
			df.to_excel(excel_writer,node_table,index = False)
			excel_writer.save()


			edge_list = get_id_flows(e_dict)
			df = pd.DataFrame(edge_list, columns = ['edge_id',ind])
			df.to_excel(excel_writer,edge_table,index = False)
			excel_writer.save()

			if df.empty:
				add_zeros_columns_to_table_psycopg2(mode_flow_tables[m], [ind],['double precision'],conn)
			else:
				df.to_sql('dummy_flows', engine, if_exists = 'replace', schema = 'public', index = False)
				add_columns_to_table_psycopg2(mode_flow_tables[m], 'dummy_flows', [ind],['double precision'], 'edge_id',conn)
			

if __name__ == '__main__':
	main()