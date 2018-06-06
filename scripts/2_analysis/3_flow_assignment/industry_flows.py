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

import transport_network_creation as tnc

import ast
from osgeo import gdal
import geopandas as gpd
from shapely.geometry import Point
from geoalchemy2 import Geometry, WKTElement

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.utils import load_config
from scripts.dbutils import *


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
mode_table = ['airport_nodes','waternodes','railnetworknodes','road2009nodes']
mode_edge_tables = ['airport_edges','wateredges','railnetworkedges','road2009edges']

mode_flow_tables = []
for mo in mode_edge_tables:
	fl_table = mo + '_flows'
	mode_flow_tables.append(fl_table)

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
od_fracs = pd.merge(od_fracs,od_data_com,how='left', on=['o','d'])

del od_data_com,od_data_modes

od_fracs = od_fracs.fillna(0)
# od_fracs.to_csv('od_fracs.csv')

for ind in ind_cols:
	'''
	Step 2 assign the crop to the closest transport mode node
	'''
	# mode_table = ['road2009nodes','railwaynetworknodes','airport_nodes','waternodes']
	# mode_edge_tables = ['road2009edges','railwaynetworkedges','airport_edges','wateredges']
	# modes = ['road','rail','air','water']

	modes = ['air','water','rail','road']
	mode_id = 'node_id' 
	od_id = 'od_id'
	pop_id = 'population'
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
	excel_writer = pd.ExcelWriter('vietnam_flow_stats_' + ind + '.xlsx')
	for m in range(len(mode_table)):
		od_nodes_regions = []
		sql_query = '''select {0}, {1}, 100*{2}/(sum({3}) over (Partition by {4})) from {5}
					'''.format(mode_id,od_id,pop_id,pop_id,od_id,mode_table[m])
		curs.execute(sql_query)
		read_layer = curs.fetchall()
		if read_layer:
			for row in read_layer:
				n = row[0]
				r = row[1]
				p = float(row[2])
				if p > 0:
					od_nodes_regions.append((n,r,p))

		all_net_dict = {'edge':[],'from_node':[],'to_node':[],'distance':[],'speed':[],'travel_cost':[]}
		all_net_dict = tnc.create_network_dictionary(all_net_dict,mode_edge_tables[m],eid,nfid,ntid,spid,'geom',curs,conn)

		od_net = tnc.create_igraph_topology(all_net_dict)

		'''
		Get the OD flows
		'''
		net_dict = {'Origin_id':[],'Destination_id':[],'Origin_region':[],'Destination_region':[],'Tonnage':[],'edge_path':[],'node_path':[]}
	
		ofile = 'network_od_flows_' + ind + modes[m] + '.csv'
		output_file = open(ofile,'w')
		wr = csv.writer(output_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		wr.writerow(net_dict.keys())

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


curs.close()
conn.close()
