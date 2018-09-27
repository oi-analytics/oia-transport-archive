# Scrape HTML details from VIWA geoserver

# URL template: http://hatang.viwa.gov.vn/BanDo/_ChiTietCangBen?id=530

import os
import sys
import re
import pandas as pd
import geopandas as gpd
import shapely

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from vtra.utils import load_config

def main():
	data_path,calc_path,output_path = load_config()['paths']['data'],load_config()['paths']['calc'],load_config()['paths']['output']

	# drop_cols = ['gid','node_id']
	# viwa_file = os.path.join(data_path,'Waterways','viwa_select','iwt_ports.shp')
	# viwa_in = gpd.read_file(viwa_file)
	# for d in drop_cols:
	# 	if d in viwa_in.columns.values.tolist():
	# 		viwa_in.drop(d,axis=1,inplace=True)

	# viwa_in['port_type'] = 'inland'

	# seaport_file = os.path.join(data_path,'Waterways','inlandandseaports','vietnam_seaport_nodes.shp')
	# seaport_in = gpd.read_file(seaport_file)

	# for d in drop_cols:
	# 	if d in seaport_in.columns.values.tolist():
	# 		seaport_in.drop(d,axis=1,inplace=True)

	# seaport_in['port_type'] = 'sea'

	# all_ports = pd.concat([viwa_in,seaport_in], axis=0, sort = 'False', ignore_index=True).fillna(0)
	# all_ports = gpd.GeoDataFrame(all_ports,crs='epsg:4326')
	# print (all_ports)

	# all_ports.to_file(os.path.join(data_path,'Waterways','waterways','ports.shp'))

	# edges_file = os.path.join(data_path,'Waterways','waterways','wateredges.shp')
	# edges_in = gpd.read_file(edges_file)
	# drop_cols = ['eid','edge_id','node_f_id','node_t_id']
	# for d in drop_cols:
	# 	if d in edges_in.columns.values.tolist():
	# 		edges_in.drop(d,axis=1,inplace=True)

	# edges_in.to_file(os.path.join(data_path,'Waterways','waterways','port_routes.shp'))

	# imp_sea_ports = {'class_1A':[135,146,147,140],'class_1':[149,150,169,175,136,137,158,163,153,151,139,171,172,170]}
	# port_ids = []
	# imp_ports_ids = [135,146,147,140,149,150,169,175,136,137,158,163,153,151,139,171,172,170]
	# for im_sp in imp_ports_ids:
	# 	if im_sp in [135,146,147,140]:
	# 		port_ids.append(('class_1A','watern_{}'.format(im_sp)))
	# 	elif im_sp in [149,150,169,175,136,137,158,163,153,151,139,171,172,170]:
	# 		port_ids.append(('class_1','watern_{}'.format(im_sp)))


	# print (port_ids)

	# ports_file = os.path.join(data_path,'Waterways','waterways','ports_nodes.shp')
	# ports_in = gpd.read_file(ports_file).fillna(0)

	# ports_in['port_class'] = 'none'
	# for p in port_ids:
	# 	ports_in.loc[ports_in['NODE_ID'] == p[1],'port_class'] = p[0]
	# 	ports_in.loc[ports_in['NODE_ID'] == p[1],'TONS'] = 5000000

	# ports_in.loc[ports_in['TONS'] == 0, 'TONS'] = 1
	# # print (ports_in)


	# ports_in.to_file(ports_file)

	air_file = os.path.join(data_path,'Airports','airnetwork','airportedges.shp')
	air_in = gpd.read_file(air_file)
	air_in['g_id'] = [int(x.split('_')[1]) for x in air_in['edge_id'].values.tolist()]
	air_in.rename(columns={'node_f_id': 'from_node'}, inplace=True)
	air_in.rename(columns={'node_t_id': 'to_node'}, inplace=True)

	air_in = air_in[['edge_id','g_id','from_node','to_node','speed','geometry']]
	air_in.to_file(os.path.join(data_path,'Airports','airnetwork','airportedges.shp'))







if __name__ == '__main__':
	main()
