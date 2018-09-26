# -*- coding: utf-8 -*-
"""
Python script to assign commodity flows on the road network in Tanzania
Created on Wed March 06 2018

@author: Raghav Pant
"""

import pandas as pd
import os
import psycopg2
import networkx as nx
import csv
import itertools
import operator
import ast
from sqlalchemy import create_engine
import numpy as np
import igraph as ig
import copy
from collections import Counter


from vtra.utils import load_config
import scripts.transport_network_creation as tnc


def get_path_commodity_od_flows(path_commodity_dataframe):
	path_commodity_dictionary = {}
	cols = path_commodity_dataframe.columns.values.tolist()
	for index, row in path_commodity_dataframe.iterrows():
		path_index = row[0]
		path_index_dict = Counter({})
		for r in range(3,len(row)):
			if str(row[r]).isdigit() is False:
				com_list = ast.literal_eval(row[r])
				for c in com_list:
					com_key = cols[r] + '_' + c[0].lower().strip() + '_' + c[1].lower().strip()
					com_val = c[2]
					com_dict = Counter({com_key:com_val})
					path_index_dict += com_dict

		path_commodity_dictionary.update({path_index:path_index_dict})

	return path_commodity_dictionary

def igraph_edge_local_disruption(network_dictionary,edge_attributes):
	fail_list = []
	for e in range(len(edge_attributes)):
		edge = edge_attributes[e][0]
		edge_tonnage = edge_attributes[e][1]
		edge_teus = edge_attributes[e][2]
		old_network = create_igraph_topology(network_dictionary,edge_tonnage,edge_teus)
		edge_cost = [n['cost'] for n in old_network.es if n['edge'] == edge][0]

		network_dictionary_copy = copy.deepcopy(network_dictionary)
		if edge in network_dictionary['edge']:
			edge_index = network_dictionary['edge'].index(edge)
			origin = network_dictionary['from_node'][edge_index]
			destination = network_dictionary['to_node'][edge_index]
			edge_distance = network_dictionary['distance'][edge_index]
			'''
			Remove the edge from the dictionary
			'''
			for key,items in network_dictionary_copy.items():
				del network_dictionary_copy[key][edge_index]

			new_network = create_igraph_topology(network_dictionary_copy,edge_tonnage,edge_teus)

			origin_node = [x for x in new_network.vs if x['node'] == origin]
			destination_node = [x for x in new_network.vs if x['node'] == destination]

			if len(origin_node) == 0 or len(destination_node) == 0:
				'''
				no alternative path exists
				'''
				new_distance = 1e20
				cost_change = edge_cost
			else:
				new_route = new_network.get_shortest_paths(origin_node[0],to = destination_node[0], weights = 'cost', mode = 'OUT', output='epath')[0]
				if new_route:
					new_distance = sum([new_network.es[n]['distance'] for n in new_route])
					cost_change = sum([new_network.es[n]['cost'] for n in new_route]) - edge_cost
				else:
					new_distance = 1e20
					cost_change = edge_cost


			distance_change = 1.0*new_distance/edge_distance

			fail_list.append((edge,edge_distance,new_distance,distance_change,edge_cost,cost_change))

	return(fail_list)

def igraph_scenario_complete_failure_old(network_dictionary,edge_failure_set,path_index_list,path_node_list,path_edge_list,path_commodity_dictionary):
	edge_fail_dictionary = Counter({})
	network_dictionary_copy = copy.deepcopy(network_dictionary)
	edge_path_index = []
	for edge in edge_failure_set:
		if edge in network_dictionary['edge']:
			edge_index = network_dictionary['edge'].index(edge)
			'''
			Remove the edge from the dictionary
			'''
			for key,items in network_dictionary_copy.items():
				del network_dictionary_copy[key][edge_index]

		edge_path_index += [e for e in range(len(path_edge_list)) if edge in path_edge_list[e]]

	new_network = create_igraph_topology(network_dictionary_copy,0,0)

	edge_path_index = list(set(edge_path_index))
	if edge_path_index:
		for e in edge_path_index:
			origin = path_node_list[e][0]
			destination = path_node_list[e][-1]

			origin_node = [x for x in new_network.vs if x['node'] == origin]
			destination_node = [x for x in new_network.vs if x['node'] == destination]

			if not origin_node or not destination_node:
				'''
				no alternative path exists
				'''
				path_index = path_index_list[e]
				edge_fail_dictionary += path_commodity_dictionary[path_index]
			else:
				new_route = new_network.get_shortest_paths(origin_node[0],to = destination_node[0], weights = 'cost', mode = 'OUT', output='epath')[0]
				if not new_route:
					'''
					no alternative path exists
					'''
					path_index = path_index_list[e]
					edge_fail_dictionary += path_commodity_dictionary[path_index]

	return edge_fail_dictionary

def igraph_scenario_complete_failure(network_dictionary,edge_failure_set,path_index_list,path_node_list,path_edge_list,path_commodity_dictionary):
	edge_fail_dictionary = Counter({})
	network_graph = tnc.create_igraph_topology(network_dictionary)
	edge_path_index = []
	for edge in edge_failure_set:
		edge_index = network_dictionary['edge'].index(edge)

		fr_nd = network_dictionary['from_node'][edge_index]
		t_nd = network_dictionary['to_node'][edge_index]
		fr_id = [x for x in network_graph.vs if x['node'] == fr_nd]
		t_id = [x for x in network_graph.vs if x['node'] == t_nd]

		network_graph.delete_edges([(fr_id[0],t_id[0])])

		edge_path_index += [e for e in range(len(path_edge_list)) if edge in path_edge_list[e]]


	edge_path_index = list(set(edge_path_index))
	if edge_path_index:
		for e in edge_path_index:
			origin = path_node_list[e][0]
			destination = path_node_list[e][-1]

			origin_node = [x for x in network_graph.vs if x['node'] == origin]
			destination_node = [x for x in network_graph.vs if x['node'] == destination]

			if not origin_node or not destination_node:
				'''
				no alternative path exists
				'''
				path_index = path_index_list[e]
				edge_fail_dictionary += path_commodity_dictionary[path_index]
			else:
				new_route = network_graph.get_shortest_paths(origin_node[0],to = destination_node[0], weights = 'distance', mode = 'OUT', output='epath')[0]
				if not new_route:
					'''
					no alternative path exists
					'''
					path_index = path_index_list[e]
					edge_fail_dictionary += path_commodity_dictionary[path_index]

	return edge_fail_dictionary

def add_columns_to_table(table_name, table_match, col_list, col_id, cursor, connection):
	for col in col_list:
		sql_query = "alter table %s add column %s double precision"%(table_name,col)
		cursor.execute(sql_query)
		connection.commit()

		# sql_query = "update %s set %s = 0"%(table_name,col)
		# cursor.execute(sql_query)
		# connection.commit()

		sql_query = '''
					update %s set %s = (select %s from %s as A where %s.%s = A.%s)
					'''%(table_name,col,col,table_match,table_name,col_id,col_id)
		cursor.execute(sql_query)
		connection.commit()

		sql_query = "update %s set %s = 0 where %s is Null"%(table_name,col,col)
		cursor.execute(sql_query)
		connection.commit()

def merge_tables(table_name, table_A,table_B,col_A,col_B,id_A,id_B,cursor, connection):
	geom_col = col_A[-1]
	sql_query = "create table %s as select "%(table_name)

	for col in col_A[:-1]:
		sql_query += "A.%s, "%(col)

	for col in col_B[1:-1]:
		sql_query += "B.%s, "%(col)

	sql_query += "B.%s, A.%s from %s as A, %s as B where A.%s = B.%s"%(col_B[-1],geom_col,table_A,table_B,id_A,id_B)
	# print (sql_query)
	cursor.execute(sql_query)
	connection.commit()

def network_od_disruption(path_index,path_list,node_path_list, net,edge,edge_fail_dict,edge_fail_dict_big):

	edge_fail_dict.update({str(edge):{'path_index':[],'old_dist':[],'new_dist':[],'incr_fact':[]}})
	path_fail_index = [item for item in range(len(path_list)) if edge in path_list[item]]
	if len(path_fail_index) >= 3000 and len(path_fail_index) < 5000:
		net_copy = net.copy()
		nodes = [(u,v) for (u,v,d) in net.edges(data = 'key') if d == edge]
		for n in nodes:
			net_copy.remove_edge(n[0],n[1])

		for index in path_fail_index:
			node_path = node_path_list[index]
			path_length = sum(get_network_edges(net, node_path,'weight'))
			if nx.has_path(net_copy,node_path[0],node_path[-1]):
				new_path_length = nx.dijkstra_path_length(net_copy,source = node_path[0],target = node_path[-1], weight = 'weight')
			else:
				new_path_length = 1e20

			path_change = 1.0*new_path_length/path_length

			edge_fail_dict[str(edge)]['path_index'].append(path_index[index])
			edge_fail_dict[str(edge)]['old_dist'].append(path_length)
			edge_fail_dict[str(edge)]['new_dist'].append(new_path_length)
			edge_fail_dict[str(edge)]['incr_fact'].append(path_change)

			# print ('Done with path %s in edge %s'%(index,edge))
	else:
		edge_fail_dict_big.update({str(edge):len(path_fail_index)})

	return(edge_fail_dict,edge_fail_dict_big)

def parse_values_string_list(string_list):
	'''
	The list is in the form of '[val1,val2,....'
	'''
	if string_list[-1] != ']':
		last_comma_find = string_list.rfind(',')
		string_list = string_list[:last_comma_find] + ']'

	value_list = ast.literal_eval(string_list)

	return (value_list)

def create_spof_info_list(all_edge_dict,file_name,sheet_name,id_col,edge_centrality_dict):
	pd_dataframe = pd.read_excel(file_name,sheet_name = sheet_name)

	edge_list = pd_dataframe[id_col].values.tolist()
	path_index_list = pd_dataframe['path_index'].values.tolist()
	old_dist_list = pd_dataframe['old_dist'].values.tolist()
	new_dist_list = pd_dataframe['new_dist'].values.tolist()
	incr_fact_list = pd_dataframe['incr_fact'].values.tolist()

	for e in range(len(edge_list)):
		edge = edge_list[e]
		edge_centrality = edge_centrality_dict[str(edge)]
		index_paths = parse_values_string_list(path_index_list[e])
		if edge_centrality > 0 and len(index_paths) > 0:
			old_paths = parse_values_string_list(old_dist_list[e])
			new_paths = parse_values_string_list(new_dist_list[e])
			incr_factors = parse_values_string_list(incr_fact_list[e])
			len_select = min(len(index_paths),len(old_paths),len(new_paths),len(incr_factors))
			all_edge_dict.update({str(edge):{'centrality':edge_centrality,'path_index':index_paths[:len_select],'old_dist':old_paths[:len_select],'new_dist':new_paths[:len_select],'incr_fact':incr_factors[:len_select]}})

		elif edge_centrality == 0 and len(index_paths) == 0:
			all_edge_dict.update({str(edge):{'centrality':0,'path_index':[],'old_dist':[],'new_dist':[],'incr_fact':[]}})

	return(all_edge_dict)




'''
Give the input fields required for establishing the database connection
'''

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

excel_writer = pd.ExcelWriter('vnm_mpof_analysis.xlsx')


mode_edge_tables = ['wateredges','railnetworkedges','road2009edges','multi_modal_edges']
eid = 'edge_id'
nfid = 'node_f_id'
ntid = 'node_t_id'
spid = 'speed'
gmid = 'geom'

all_net_dict = {'edge':[],'from_node':[],'to_node':[],'distance':[],'speed':[],'travel_cost':[]}

for m in range(len(mode_edge_tables)):
	all_net_dict = tnc.create_network_dictionary(all_net_dict,mode_edge_tables[m],eid,nfid,ntid,spid,'geom',cur,conn)
	print('done with',mode_edge_tables[m])

# od_net = tnc.create_igraph_topology(all_net_dict)


'''
Path OD flow disruptions
'''
df = pd.read_excel("vnm_path_flows.xlsx",sheet_name = 'path_flows')
pth_key_list = df['path_index'].values.tolist()

npth_list = df['node_path'].values.tolist()
npth_list = [ast.literal_eval(npaths) for npaths in npth_list]

epth_list = df['edge_path'].values.tolist()
epth_list = [ast.literal_eval(epaths) for epaths in epth_list]

pth_com_dict = get_path_commodity_od_flows(df)


col_names = ['level13_vt_100_mask_1','level14_vt_100_mask_1','level15_vt_100_mask_1','level16_vt_100_mask_1']

ef_list = []
for rp in col_names:
	fl_df = pd.read_excel("vnm_road_rail_flood_list.xlsx",sheet_name = rp)
	fl_list = list(fl_df.itertuples(index=False))
	for region,scenario in fl_list:
		sc = ast.literal_eval(scenario)
		if sc['flood_edge']:
			ef_set = sc['flood_edge']
			ef_dict = igraph_scenario_complete_failure(all_net_dict,ef_set,pth_key_list,npth_list,epth_list,pth_com_dict)
			if ef_dict:
				for key,values in ef_dict.items():
					key_vals = key.split('_')
					ind = key_vals[0]
					fr = key_vals[1]
					tr = key_vals[2]
					ef_list.append((region,rp,ef_set,ind,fr,tr,values))

		print ('Done with region {0} in coastal flooding {1}'.format(region,rp))

df = pd.DataFrame(ef_list,columns = ['region_flooded','return_period','senario_edges','industry','from_region','to_region','tonnage'])
df.to_csv('vnm_road_rail_edge_multi_failure.csv',index = False)


# e_list = [x[0] for x in road_link_attr] + rail_link_attr

# # print (epth_list)
# ef_dict = {}
# ef_dict_big = {}
# for e in e_list:
# 	ef_dict,ef_dict_big = network_od_disruption(pth_key_list,epth_list,npth_list,od_net,e,ef_dict,ef_dict_big)
# 	print ('Done with edge:',e)

# fail_tuple_list = []
# for key, values in ef_dict.items():
# 	if key.isdigit():
# 		eid = int(key)
# 	else:
# 		eid = key


# 	fail_paths = ef_dict[key]['path_index']
# 	old_lengths = ef_dict[key]['old_dist']
# 	new_lengths = ef_dict[key]['new_dist']
# 	length_changes = ef_dict[key]['incr_fact']

# 	fail_tuple_list.append((eid,fail_paths,old_lengths,new_lengths,length_changes))


# road_fail_list = [x for x in fail_tuple_list if str(x[0]).isdigit()]
# rail_fail_list = [x for x in fail_tuple_list if 'rail' in str(x[0])]

# excel_writer = pd.ExcelWriter('tz_spof_analysis_9.xlsx')

# cols = ['link','path_index','old_dist','new_dist','incr_fact']
# df = pd.DataFrame(road_fail_list, columns = cols)
# df.to_excel(excel_writer,'road_od_disrupt', index = False)
# excel_writer.save()

# cols = ['id','path_index','old_dist','new_dist','incr_fact']
# df = pd.DataFrame(rail_fail_list, columns = cols)
# df.to_excel(excel_writer,'rail_od_disrupt', index = False)
# excel_writer.save()

# big_fail_tuple_list = []
# for key, values in ef_dict_big.items():
# 	if key.isdigit():
# 		eid = int(key)
# 	else:
# 		eid = key

# 	big_fail_tuple_list.append((eid,values))

# cols = ['edge_id','fail_paths_no']
# df = pd.DataFrame(big_fail_tuple_list, columns = cols)
# df.to_excel(excel_writer,'big_disrupt', index = False)
# excel_writer.save()

'''
Process results
'''
# file_no = [4,5,6,7,8,9]
# all_edges = []
# for f in file_no:
# 	f_name = 'tz_spof_analysis_' + str(f) + '.xlsx'
# 	df = pd.read_excel(f_name,sheet_name = 'big_disrupt')
# 	e_list = df['edge_id'].values.tolist()
# 	pth_nos = df['fail_paths_no'].values.tolist()
# 	e_info = list(zip(e_list,pth_nos))
# 	all_edges += e_info

# # print (all_edges)
# unique_edges = list(set([x[0] for x in all_edges]))
# e_cntr_dict = {}
# # print (unique_edges)
# for u in unique_edges:
# 	upth_nos = [x[1] for x in all_edges if x[0] == u]
# 	# print (u,upth_nos)
# 	e_cntr_dict.update({str(u):upth_nos[0]})

# # print (e_cntr_dict)

# all_e_dict = {}
# for f in file_no:
# 	f_name = 'tz_spof_analysis_' + str(f) + '.xlsx'
# 	all_e_dict = create_spof_info_list(all_e_dict,f_name,'road_od_disrupt','link',e_cntr_dict)
# 	all_e_dict = create_spof_info_list(all_e_dict,f_name,'rail_od_disrupt','id',e_cntr_dict)

# # print (all_e_dict)

# pth_flow_list = []
# df = pd.read_excel('tz_path_flows.xlsx',sheet_name = 'path_flows')
# cols = df.columns.values.tolist()
# industry_names = cols[3:]
# # print (industry_names)
# for idx, r in df.iterrows():
# 	pth_flow = []
# 	pth_flow.append(r[0])
# 	for i in range(3,len(r)):
# 		pth_flow.append(r[i])

# 	pth_flow_list.append(pth_flow)

# total_pths = len(pth_flow_list)

# all_e_tuple = []
# for key,values in all_e_dict.items():
# 	if key.isdigit():
# 		e_id = int(key)
# 	else:
# 		e_id = key

# 	e_cntr = 1.0*all_e_dict[key]['centrality']/total_pths
# 	pth_idx_list = all_e_dict[key]['path_index']
# 	o_dst_list = all_e_dict[key]['old_dist']
# 	n_dst_list = all_e_dict[key]['new_dist']
# 	incr_dst_list = all_e_dict[key]['incr_fact']

# 	ind_flow_vals = np.array([0]*len(industry_names), dtype = np.float64)
# 	# print (ind_flow_vals)
# 	ton_km = 0
# 	low_day_trpt_loss = 0.0
# 	high_day_trpt_loss = 0.0
# 	for p in range(len(pth_idx_list)):
# 		o_dst = o_dst_list[p]
# 		n_dst = n_dst_list[p]
# 		incr_dst = incr_dst_list[p]

# 		pth_flow_vals = [pflow for pflow in pth_flow_list if pflow[0] == pth_idx_list[p]]
# 		pth_flow_vals = pth_flow_vals[0]
# 		if incr_dst > 1e5:
# 			# print (pth_flow_vals)
# 			ind_flow_vals += np.array(pth_flow_vals[1:])
# 			ton_km += o_dst*pth_flow_vals[-1]
# 		else:
# 			ton_km += (n_dst - o_dst)*pth_flow_vals[-1]

# 	if ton_km > 0:
# 		low_day_trpt_loss = 0.12*ton_km/365.0
# 		high_day_trpt_loss = 0.15*ton_km/365.0


# 	e_tuple = [e_id,e_cntr,ton_km,low_day_trpt_loss,high_day_trpt_loss] + list(ind_flow_vals)
# 	all_e_tuple.append(e_tuple)


# road_list = [e for e in all_e_tuple if str(e[0]).isdigit()]
# rail_list = [e for e in all_e_tuple if 'rail' in str(e[0])]

# excel_writer = pd.ExcelWriter('tz_spof_flow_analysis.xlsx')

# df = pd.DataFrame(road_list, columns = ['link','centrality','ton_km_loss','tr_p_incr_low','tr_p_incr_high'] + industry_names)
# df.to_excel(excel_writer,'road_od_losses',index = False)
# excel_writer.save()


# df = pd.DataFrame(rail_list, columns = ['id','centrality','ton_km_loss','tr_p_incr_low','tr_p_incr_high'] + industry_names)
# df.to_excel(excel_writer,'rail_od_losses',index = False)
# excel_writer.save()
