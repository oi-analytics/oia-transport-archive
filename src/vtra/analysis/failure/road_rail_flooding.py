
# -*- coding: utf-8 -*-
"""
Python script to assign commodity flows on the road network in Tanzania
Created on Wed Nov 26 2017

@author: Raghav Pant
"""

import pandas as pd
import os
import psycopg2
import networkx as nx
import csv
from sqlalchemy import create_engine
import subprocess as sp
import operator
import itertools
import copy
import matplotlib.pyplot as plt


from vtra.utils import load_config

def add_column_to_table(table_name, table_match, col_name, col_id, cursor, connection):
	sql_query = "alter table %s add column %s double precision"%(table_name,col_name)
	cursor.execute(sql_query)
	connection.commit()


	sql_query = '''
				update %s set %s = (select %s from %s as A where %s.%s = A.%s)
				'''%(table_name,col_name,col_name,table_match,table_name,col_id,col_id)
	cursor.execute(sql_query)
	connection.commit()

	sql_query = "update %s set %s = 0 where %s is Null"%(table_name,col_name,col_name)
	cursor.execute(sql_query)
	connection.commit()

def add_column_value_to_table(table_name, col_name, col_id,col_id_val_list, cursor, connection):
	sql_query = "alter table %s add column %s integer"%(table_name,col_name)
	cursor.execute(sql_query)
	connection.commit()

	for id_val in col_id_val_list:
		c_id = id_val[0]
		c_val = id_val[1]
		if type(c_id) is str:
			sql_query = '''
						update %s set %s = %s where %s = '%s'
						'''%(table_name,col_name,c_val,col_id,c_id)
		else:
			sql_query = '''
						update %s set %s = %s where %s = %s
						'''%(table_name,col_name,c_val,col_id,c_id)

		cursor.execute(sql_query)
		connection.commit()

	sql_query = "update %s set %s = 0 where %s is Null"%(table_name,col_name,col_name)
	cursor.execute(sql_query)
	connection.commit()

def get_total_edges(sector_region_list):
	region_edges_list = []
	sector_region_dict = {}
	for sid,region in sector_region_list:
		if region not in sector_region_dict.keys():
			sector_region_dict.update({region:[sid]})
		else:
			sector_region_dict[region].append(sid)

	for region,edges in sector_region_dict.items():
		total_edges = len(edges)

		region_edges_list.append([region,total_edges])

	return region_edges_list

def get_flood_results(flood_list,element_type,sector_type,model_include,flood_depth_min,flood_depth_max,flood_return_period,sector_list,sector_region_list):
	flood_percentage_list = []
	all_flood_edges = [fr.id for fr in flood_list if fr.network_element == element_type
					and fr.sector == sector_type and fr.model in model_include and fr.flood_depth >= flood_depth_min
					and fr.flood_depth < flood_depth_max and fr.return_period == flood_return_period]
	all_flood_edges = list(set(all_flood_edges))
	# print (all_flood_edges)
	if sector_type == 'road':
		selected_flood_edges = [int(fr) for fr in all_flood_edges if int(fr) in sector_list]
	else:
		selected_flood_edges = [fr for fr in all_flood_edges if fr in sector_list]

	selected_flood_edges = list(set(selected_flood_edges))
	percent_sector_flooded = int(100.0*len(selected_flood_edges)/len(sector_list))
	# flood_percentage_list.append(('Total',percent_sector_flooded))
	flood_percentage_list.append(('Total',len(selected_flood_edges)))


	sector_region_dict = {}
	for sid,region in sector_region_list:
		if region not in sector_region_dict.keys():
			sector_region_dict.update({region:{'edge':[sid],'flood_edge':[]}})
			if sid in selected_flood_edges:
				sector_region_dict[region]['flood_edge'].append(sid)
		else:
			sector_region_dict[region]['edge'].append(sid)
			if sid in selected_flood_edges:
				sector_region_dict[region]['flood_edge'].append(sid)

	for region,edges in sector_region_dict.items():
		total_edges = len(edges['edge'])
		flooded_edges = len(edges['flood_edge'])

		if total_edges > 0:
			percent_flood = int(100.0*flooded_edges/total_edges)
		else:
			percent_flood = 0

		# flood_percentage_list.append((region,percent_flood))
		flood_percentage_list.append((region,flooded_edges))


	return flood_percentage_list

def get_flood_id_depth(flood_list,element_type,sector_type,model_include,flood_depth_threshold_min,flood_depth_threshold_max,flood_return_period,sector_list,sector_region_list):
	sector_region_dict = {}
	all_flood_edges = [(fr.id,fr.flood_depth) for fr in flood_list if fr.network_element == element_type
					and fr.sector == sector_type and fr.model in model_include and fr.flood_depth >= flood_depth_threshold_min
					and fr.flood_depth < flood_depth_threshold_max and fr.return_period == flood_return_period]
	all_flood_edges = list(set(all_flood_edges))
	# print (all_flood_edges)
	if sector_type == 'road':
		selected_flood_edges = [(int(fr[0]),fr[1]) for fr in all_flood_edges if int(fr[0]) in sector_list]
	else:
		selected_flood_edges = [fr for fr in all_flood_edges if fr[0] in sector_list]

	# selected_flood_edges = [fr for fr in all_flood_edges if fr[0] in sector_list]
	# percent_sector_flooded = int(100.0*len(selected_flood_edges)/len(sector_list))
	# flood_percentage_list.append(('Total',percent_sector_flooded))
	# flood_percentage_list.append(('Total',len(selected_flood_edges)))


	sector_region_dict = {}
	for sid,region in sector_region_list:
		if region not in sector_region_dict.keys():
			sector_region_dict.update({region:{'flood_edge':[]}})
			if sid in [s[0] for s in selected_flood_edges]:
				fattr = [s for s in selected_flood_edges if s[0] == sid]
				if fattr:
					fattr_max = max([fa[1] for fa in fattr])
					fattr = [sid,fattr_max]
				sector_region_dict[region]['flood_edge'].append(fattr)

		elif sid in [s[0] for s in selected_flood_edges]:
				fattr = [s for s in selected_flood_edges if s[0] == sid]
				if fattr:
					fattr_max = max([fa[1] for fa in fattr])
					fattr = [sid,fattr_max]
				sector_region_dict[region]['flood_edge'].append(fattr)


	return sector_region_dict

def main():
	curdir = os.getcwd()

	conf = load_config()

	try:
		conn = psycopg2.connect(**conf['database'])
	except:
		print ("I am unable to connect to the database")


	curs = conn.cursor()

	engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format({
		**conf['database']
	}))


	layers_ids = [('roads_edge_flood','edge_id','road2009edges','geom'),
				('rail_edge_flood','edge_id','railnetworkedges','geom'),
				('roads_node_flood','node_id','road2009nodes','geom'),
				('rail_node_flood','node_id','railnetworknodes','geom'),
				('port_flood','node_id','seaport_nodes','geom'),
				('airport_flood','node_id','airport_nodes','geom'),
				('provinces_flooding','name_eng','province_level_stats','geom')]

	regional_table = 'province_level_stats'
	mode_edge_tables = ['rail_edge_flood','roads_edge_flood']
	edge_id = 'edge_id'
	regional_name = 'name_eng'
	col_names = ['level13_vt_100_mask_1','level14_vt_100_mask_1','level15_vt_100_mask_1','level16_vt_100_mask_1']

	excel_writer = pd.ExcelWriter('vnm_road_rail_flood_list.xlsx')


	for c in col_names:
		fl_rg_list = []
		for m in mode_edge_tables:
			edge_region_list = []
			sql_query = '''SELECT A.{0},B.{1} FROM {2} as A, {3} as B where st_intersects(A.geom,B.geom) is True and A.{4} > 0
						'''.format(edge_id,regional_name,m,regional_table,c)
			cur.execute(sql_query)
			read_layer = cur.fetchall()
			for row in read_layer:
				edge_region_list.append((row[0],row[1]))

			tedge_regions = [r[1] for r in edge_region_list]
			tedge_regions = sorted(list(set(tedge_regions)))

			for rg in tedge_regions:
				ed = [r[0] for r in edge_region_list if r[1] == rg]
				fl_rg_list.append((rg,{'flood_edge':ed}))

		fl_rg_list = sorted(fl_rg_list, key=lambda x: x[0])
		df = pd.DataFrame(fl_rg_list,columns = ['region','flood_list'])
		df.to_excel(excel_writer,c,index = False)
		excel_writer.save()


	conf = load_config()

	try:
		conn = psycopg2.connect(**conf['database'])
	except:
		print ("I am unable to connect to the database")


	curs = conn.cursor()

	engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format({
		**conf['database']
	}))


	layers_ids = [('roads_edge_flood','edge_id','road2009edges','geom'),
				('rail_edge_flood','edge_id','railnetworkedges','geom'),
				('roads_node_flood','node_id','road2009nodes','geom'),
				('rail_node_flood','node_id','railnetworknodes','geom'),
				('port_flood','node_id','seaport_nodes','geom'),
				('airport_flood','node_id','airport_nodes','geom'),
				('provinces_flooding','name_eng','province_level_stats','geom')]

	regional_table = 'province_level_stats'
	mode_edge_tables = ['rail_edge_flood','roads_edge_flood']
	edge_id = 'edge_id'
	regional_name = 'name_eng'
	col_names = ['level13_vt_100_mask_1','level14_vt_100_mask_1','level15_vt_100_mask_1','level16_vt_100_mask_1']

	excel_writer = pd.ExcelWriter('vnm_road_rail_flood_list.xlsx')


	for c in col_names:
		fl_rg_list = []
		for m in mode_edge_tables:
			edge_region_list = []
			sql_query = '''SELECT A.{0},B.{1} FROM {2} as A, {3} as B where st_intersects(A.geom,B.geom) is True and A.{4} > 0
						'''.format(edge_id,regional_name,m,regional_table,c)
			cur.execute(sql_query)
			read_layer = cur.fetchall()
			for row in read_layer:
				edge_region_list.append((row[0],row[1]))

			tedge_regions = [r[1] for r in edge_region_list]
			tedge_regions = sorted(list(set(tedge_regions)))

			for rg in tedge_regions:
				ed = [r[0] for r in edge_region_list if r[1] == rg]
				fl_rg_list.append((rg,{'flood_edge':ed}))

		fl_rg_list = sorted(fl_rg_list, key=lambda x: x[0])
		df = pd.DataFrame(fl_rg_list,columns = ['region','flood_list'])
		df.to_excel(excel_writer,c,index = False)
		excel_writer.save()

if __name__ == '__main__':
	main()
