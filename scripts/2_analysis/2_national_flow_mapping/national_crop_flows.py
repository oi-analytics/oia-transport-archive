"""Summarise hazard data

Get OD data and process it
Author: Raghav Pant
Date: April 20, 2018
"""
import configparser
import csv
import glob
import os

import fiona
import fiona.crs
import rasterio


from sqlalchemy import create_engine
import subprocess as sp
import psycopg2
import osgeo.ogr as ogr

import pandas as pd
import copy

import ast
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Point
from geoalchemy2 import Geometry, WKTElement

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.utils import load_config
from scripts.dbutils import *
import scripts.transport_network_creation as tnc


'''
Create the database connection
'''
conf = load_config()

try:
	conn = psycopg2.connect(**conf['database'])
except:
	print ("I am unable to connect to the database")


curs = conn.cursor()

engine = create_engine('postgresql://{user}:{password}@{host}:{port}/{database}'.format({
	**conf['database']
}))

crop_data_path = os.path.join(conf['paths']['data'], 'crop_data')
od_data_file = os.path.join(conf['paths']['data'], 'od_data', 'OD_transport_data_2008_v2.xlsx')
'''
Step 2: Create the OD proprotions for the differnet modes
'''
'''
First get the modal shares
'''
modes = ['road','rail','air','water']
mode_cols = ['road','rail','air','inland','coastal']
new_mode_cols = ['o','d','road','rail','air','water']
crop_names = ['rice','cash','cass','teas','maiz','rubb','swpo','acof','rcof','pepp']
mode_table = ['airport_nodes','waternodes','railnetworknodes','road2009nodes']
mode_edge_tables = ['airport_edges','wateredges','railnetworkedges','road2009edges']

mode_flow_tables = []
for mo in mode_edge_tables:
	fl_table = mo + '_flows'
	mode_flow_tables.append(fl_table)
	drop_postgres_table_psycopg2(fl_table,conn)

	with conn.cursor() as cur:
		sql_query = "create table {0} as select edge_id,geom from {1}".format(fl_table,mo)
		cur.execute(sql_query)
		conn.commit()


'''
Get the modal shares
'''
od_data_modes = pd.read_excel(
	od_data_file,
	sheet_name = 'mode'
	).fillna(0)
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
crop_cols = ['rice','indust-cro']
for cr in crop_cols:
	od_data_com_sums = od_data_com.groupby(['o','d']).agg({cr: 'sum'})
	od_com_frac = od_data_com_sums.groupby(level=0).apply(lambda x: x/float(x.sum()))
	od_com_frac = od_com_frac.reset_index(level=['o', 'd'])
	od_fracs = pd.merge(od_fracs,od_com_frac,how='left', on=['o','d'])

del od_data_com,od_data_com_sums,od_com_frac

od_fracs = od_fracs.fillna(0)
# od_fracs.to_csv('od_fracs.csv')

for file in os.listdir(crop_data_path):
	if file.endswith(".tif") and 'spam_p' in file.lower().strip():
		fpath = os.path.join(crop_data_path, file)
		crop_name = [cr for cr in crop_names if cr in file.lower().strip()][0]
		raster_in = fpath
		outCSVName = 'crop_concentrations.csv'
		crop_table = crop_name + '_production'
		'''Clip to region and convert to points'''
		os.system('gdal2xyz.py -csv '+raster_in+' '+ outCSVName)

		'''Load points and convert to geodataframe with coordinates'''    
		load_points = pd.read_csv(outCSVName,header=None,names=['x','y','crop_prod'],index_col=None)
		load_points = load_points[load_points['crop_prod'] > 0]
		# load_points.to_csv('crop_concentrations.csv', index = False)

		geometry = [Point(xy) for xy in zip(load_points.x, load_points.y)]
		load_points = load_points.drop(['x', 'y'], axis=1)
		crs = {'init': 'epsg:4326'}
		points_gdp = gpd.GeoDataFrame(load_points, crs=crs, geometry=geometry)
		points_gdp['geom'] = points_gdp['geometry'].apply(lambda x: WKTElement(x.wkt, srid=4326))

		#drop the geometry column as it is now duplicative
		points_gdp.drop('geometry', 1, inplace=True)
		# points_gdp = points_gdp.rename(columns={'geometry':'geom'}).set_geometry('geom')
		del load_points

		print ('created geopandas dataframe from the points')

		points_gdp.to_sql(crop_table, engine, if_exists = 'replace', schema = 'public', index = True,dtype={'geom': Geometry('POINT', srid= 4326)})

		del points_gdp
		'''
		Add gid field to crop table
		'''
		with conn.cursor() as cur:
			sql_query = "alter table {0} add column gid serial".format(crop_table)
			cur.execute(sql_query)
			conn.commit()

		print ('Done with loading crop table to database')

		'''
		Step 1
		Assign the regions closest to the crop nodes
		'''
		nd_table = crop_table
		regional_table = 'province_level_stats'
		dummy_table = 'dummy_table'
		nd_id = 'gid'
		nd_gm = 'geom'
		regional_gm = 'geom'

		nd_attr = ['gid']
		regional_attr = ['name_eng','provinceid','od_id']
		regional_attr_type = ['character varying', 'integer','integer']

		drop_postgres_table_psycopg2(dummy_table,conn)
		# nodes_polygons_nearest_psycopg2(dummy_table,nd_table,regional_table,nd_attr,regional_attr,nd_gm,regional_gm,cur,conn)
		nodes_polygons_within_nearest_psycopg2(dummy_table,nd_table,regional_table,nd_attr,nd_id,regional_attr,nd_gm,regional_gm,conn)
		print ('Done with assigning attributes to the crop table')

		add_columns_to_table_psycopg2(nd_table,dummy_table,regional_attr,regional_attr_type,nd_id,conn)
		print ('Done with adding columns to the crop table')


		'''
		Step 2 assign the crop to the closest transport mode node
		'''
		# mode_table = ['road2009nodes','railwaynetworknodes','airport_nodes','waternodes']
		# mode_edge_tables = ['road2009edges','railwaynetworkedges','airport_edges','wateredges']
		# modes = ['road','rail','air','water']

		modes = ['air','water','rail','road']
		mode_id = 'node_id'
		crop_id = 'gid'
		mode_crop_m = 'od_id'
		crop_mode_m = 'od_id'
		crop_prod = 'crop_prod' 
		od_id = 'od_id'
		od_id_type = 'integer'
		o_id_col = 'o'
		d_id_col = 'd'

		'''
		Get the network
		'''
		eid = 'edge_id'
		nfid = 'node_f_id'
		ntid = 'node_t_id'
		spid = 'speed'
		gmid = 'geom'
		o_id_col = 'o'
		d_id_col = 'd'

		'''
		Get the node edge flows
		'''
		excel_writer = pd.ExcelWriter('vietnam_flow_stats_' + crop_name + '.xlsx')
		for m in range(len(mode_table)):
			drop_postgres_table_psycopg2(dummy_table,conn)
			nodes_polygons_aggregations(dummy_table,mode_table[m],crop_table,mode_id,crop_id,mode_crop_m,crop_mode_m,crop_prod,nd_gm,regional_gm,conn)
			add_columns_to_table_psycopg2(dummy_table,mode_table[m],[od_id],[od_id_type],mode_id,conn)

			od_nodes_regions = []

			with conn.cursor() as cur:
				sql_query = '''select {0}, {1}, {2}, {3}/(sum({4}) over (Partition by {5})) from {6}
							'''.format(mode_id,od_id,crop_prod,crop_prod,crop_prod,od_id,dummy_table)
				cur.execute(sql_query)
				read_layer = cur.fetchall()
				if read_layer:
					for row in read_layer:
						n = row[0]
						r = row[1]
						c = float(row[2])
						p = float(row[3])
						if p > 0:
							od_nodes_regions.append((n,r,c,p))


			all_net_dict = {'edge':[],'from_node':[],'to_node':[],'distance':[],'speed':[],'travel_cost':[]}
			all_net_dict = tnc.create_network_dictionary(all_net_dict,mode_edge_tables[m],eid,nfid,ntid,spid,'geom',curs,conn)

			od_net = tnc.create_igraph_topology(all_net_dict)

			'''
			Get the OD flows
			'''
			net_dict = {'Origin_id':[],'Destination_id':[],'Origin_region':[],'Destination_region':[],'Tonnage':[],'edge_path':[],'node_path':[]}
	
			ofile = 'network_od_flows_' + crop_name + modes[m] + '.csv'
			output_file = open(ofile,'w')
			wr = csv.writer(output_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			wr.writerow(net_dict.keys())

			crop_mode = modes[m]+ '_' + crop_name
			if crop_name in ('rice', 'cereal', 'wheat'):
				od_fracs[crop_mode] = od_fracs[modes[m]]*od_fracs['rice']
			else:
				od_fracs[crop_mode] = od_fracs[modes[m]]*od_fracs['indust-cro']

			od_flows = list(zip(od_fracs[o_id_col].values.tolist(),od_fracs[d_id_col].values.tolist(),od_fracs[crop_mode].values.tolist()))
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
								o_val = 1.0*fval[0]*o_vals[1]
								o_node = o_vals[0]
								d_matches = [(item[0],item[3]) for item in od_nodes_regions if item[1] == d]
								if len(d_matches) > 0:
									for d_vals in d_matches:
										od_val = 1.0*o_val*d_vals[1]/365
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
									

					print (o,d,fval,modes[m],crop_name)


			node_table = modes[m] + '_node_flows'
			edge_table = modes[m] + '_edge_flows'

			# dom_flows = pd.read_csv(ofile).fillna(0)
			dom_flows = pd.DataFrame(dflows,columns = ['node_path', 'edge_path','Tonnage'])

			flow_node_edge = dom_flows.groupby(['node_path', 'edge_path'])['Tonnage'].sum().reset_index()
			n_dict = {}
			e_dict = {}
			n_dict,e_dict = get_node_edge_flows(flow_node_edge,n_dict,e_dict)

			node_list = get_id_flows(n_dict)
			df = pd.DataFrame(node_list, columns = ['node_id',crop_name])
			df.to_excel(excel_writer,node_table,index = False)
			excel_writer.save()


			edge_list = get_id_flows(e_dict)
			df = pd.DataFrame(edge_list, columns = ['edge_id',crop_name])
			df.to_excel(excel_writer,edge_table,index = False)
			excel_writer.save()

			if df.empty:
				add_zeros_columns_to_table_psycopg2(mode_flow_tables[m], [crop_name],['double precision'],conn)
			else:
				df.to_sql('dummy_flows', engine, if_exists = 'replace', schema = 'public', index = False)
				add_columns_to_table_psycopg2(mode_flow_tables[m], 'dummy_flows', [crop_name],['double precision'], 'edge_id',conn)


curs.close()
conn.close()
