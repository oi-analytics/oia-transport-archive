"""
Get vietnam shapefiles and convert them into networks
@author: Raghav Pant
Date: June 25, 2018
"""
import os
import sys

from vtra.utils import *
import vtra.preprocess.network_create as nc

def check_single_linegeom_creation(line_table,line_id,line_geom_type,line_attr,line_attr_typ):
	if line_geom_type == 'multilinestring':
		sg_ln_table = nc.convert_multiline_to_linestring(line_table,'gid','geom',line_attr,line_attr_typ)
		sg_ln_id = 'lineid'
	else:
		sg_ln_table = line_table
		sg_ln_id = line_id

	return sg_ln_table, sg_ln_id

def main():
	# sectors = ['Road','Railways','Marine','InlandWaterways']
	# sector_ids = ['road','rail','sea','iwt']
	# subsectors = [['national_roads','laocai_roads','binhdinh_roads','thanhhoa_roads'],['national_rail'],['marine'],['viwa']]
	# cm_attr = [['','','',''],['railwaylin'],['ma'],['']]

	config = load_config()
	# sectors = ['Roads']
	# sector_ids = ['road']
	# subsectors = [['national_roads','laocai_roads','binhdinh_roads','thanhhoa_roads']]
	# sub_enc = [['latin1','utf-8','utf-8','utf-8']]
	# cm_attr = [['','','','']]

	sectors = ['Railways']
	sector_ids = ['rail']
	subsectors = [['national_rail']]
	sub_enc = [['utf-8']]
	cm_attr = [['railwaylin']]
	cm_attr_typ = [['character varying']]

	# sectors = ['Roads','Railways']
	# sector_ids = ['road','rail']
	# subsectors = [['national_roads'],['national_rail']]
	# sub_enc = [['latin1'],['utf-8']]
	# cm_attr = [[''],['railwaylin']]
	# cm_attr_typ = [[''],['character varying']]

	# sectors = ['Roads']
	# sector_ids = ['road']
	# subsectors = [['laocai_roads','binhdinh_roads','thanhhoa_roads']]
	# sub_enc = [['utf-8','utf-8','utf-8']]
	# cm_attr = [['','','']]
	# cm_attr_typ = [['','','']]

	# pt_id = 'gid'
	# ln_id = 'gid'
	# pt_gm = 'geom'
	# ln_gm = 'geom'
	# dst_thr = 100
	# dst_prox = 20

	# nd_id = 'node_id'
	# edg_id = 'edge_id'
	# edg_int_id = 'g_id'
	# f_nd = 'from_node'
	# t_nd = 'to_node'
	# nd_gm = 'geom'
	# edg_gm = 'geom'
	# nd_prox = 0

	# sectors = ['Roads']
	# sector_ids = ['road']
	# subsectors = [['national_network_edges_v1']]
	# sectors = ['Waterways']
	# sector_ids = ['water']
	# subsectors = [['waterways']]
	# sectors = ['Air']
	# sector_ids = ['air']
	# subsectors = [['air']]
	# sub_enc = [['utf-8']]
	# cm_attr = [['']]
	# cm_attr_typ = [['']]

	pt_id = 'gid'
	ln_id = 'gid'
	pt_gm = 'geom'
	ln_gm = 'geom'
	dst_thr = 100
	dst_prox = 20

	nd_id = 'node_id'
	edg_id = 'edge_id'
	edg_int_id = 'g_id'
	f_nd = 'from_node'
	t_nd = 'to_node'
	nd_gm = 'geom'
	edg_gm = 'geom'
	nd_prox = 20
	skip_ids = ['gid','edge_id','from_node','to_node','node_id','geom']

	for s in range(len(sectors)):
		sect = sectors[s]
		for sb_list in range(len(subsectors[s])):
			subsect = subsectors[s][sb_list]
			input_path = os.path.join(config['paths']['data'],'infrastructure_preprocessed_data',sect,subsect)
			pt_table, ln_table, pt_gm_typ, ln_gm_typ = nc.write_shapefiles_to_database(input_path,sub_enc[s][sb_list])
			print (pt_table, ln_table, pt_gm_typ, ln_gm_typ)
			# pt_table = ''
			# ln_table = 'national_network_edges_v1'
			# pt_gm_typ= ''
			# ln_gm_typ = 'multilinestring'
			nd_table, edg_table = nc.create_node_edge_tables_from_point_line_tables(pt_table,ln_table)
			if pt_table:
				# We have a point file and a line file
				sln_table, sln_id = check_single_linegeom_creation(ln_table,ln_id,ln_gm_typ,cm_attr[s][sb_list],cm_attr_typ[s][sb_list])
				print ("Done with geometry check")
				# print (sln_table, sln_id)
				# print (pt_table,sln_table,pt_id,ln_id,cm_attr[s][sb_list],cm_attr[s][sb_list],pt_gm,ln_gm,dst_thr,dst_prox)
				pt_ln_list = nc.match_points_to_lines(pt_table,sln_table,pt_id,ln_id,cm_attr[s][sb_list],cm_attr[s][sb_list],pt_gm,ln_gm,dst_thr,dst_prox)
				# print (pt_ln_list)
				# print (list(set([p[1] for p in pt_ln_list])))
				print ("Done with matching points and lines")
				# print (pt_table,sln_table,pt_id,ln_id,sln_id,pt_gm,ln_gm,pt_ln_list,sector_ids[s],dst_thr,nd_table,edg_table)
				nc.insert_to_node_edge_tables_from_given_points_lines(pt_table,sln_table,pt_id,ln_id,sln_id,pt_gm,ln_gm,pt_ln_list,sector_ids[s],dst_thr,nd_table,edg_table)
				print ("Done with inserting nodes and edges")
				nc.eliminate_common_nodes_from_network(nd_table,edg_table,nd_id,nd_gm,nd_prox)
				print ("Done with eliminating common nodes")
				nc.bisect_lines_by_nodes(nd_table,edg_table,nd_id,edg_id,edg_int_id,f_nd,t_nd,ln_id,nd_gm,edg_gm,nd_prox)
				print ("Done with line bisection by nodes")
				nc.add_all_columns_from_one_table_to_another(ln_table,edg_table,ln_id,skip_ids)
				nc.add_all_columns_from_one_table_to_another(pt_table,nd_table,pt_id,skip_ids)
				print ("Done with adding columns")
			else:
				# We only have a line file from which we create node and edge tables
				sln_table,sln_id = check_single_linegeom_creation(ln_table,ln_id,ln_gm_typ,cm_attr[s][sb_list],cm_attr_typ[s][sb_list])
				print ("Done with geometry check")
				nc.insert_to_node_edge_tables_from_line_table(sector_ids[s],nd_table,edg_table,sln_table,sln_id,ln_gm)
				print ("Done with inserting nodes and edges")
				nc.eliminate_common_nodes_from_network(nd_table,edg_table,nd_id,nd_gm,nd_prox)
				print ("Done with eliminating common nodes")
				nc.bisect_lines_by_nodes(nd_table,edg_table,nd_id,edg_id,edg_int_id,f_nd,t_nd,ln_id,nd_gm,edg_gm,nd_prox)
				print ("Done with line bisection by nodes")
				nc.add_all_columns_from_one_table_to_another(ln_table,edg_table,ln_id,skip_ids)
				print ("Done with adding columns")

			output_path = os.path.join(config['paths']['data'],'infrastructure_processed_data',sect,subsect)
			nc.export_pgsql_to_shapefiles(output_path,nd_table)
			nc.export_pgsql_to_shapefiles(output_path,edg_table)
			print ("Done with exporting nodes and edges to shapefiles")

if __name__ == '__main__':
    main()
