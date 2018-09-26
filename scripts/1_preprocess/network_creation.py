"""Summarise hazard data

Get a shapefile and convert it into a network
Author: Raghav Pant
Date: April 17, 2018
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


from vtra.utils import load_config

def extract_points(gm):
	x_pts = []
	y_pts = []
	for k in range(0, gm.GetPointCount()):
		pt = gm.GetPoint(k)
		x_pts.append(pt[0])
		y_pts.append(pt[1])

	return x_pts, y_pts


def get_geom_points(gtext):
	x_list = []
	y_list = []
	gtype = ogr.CreateGeometryFromWkt(gtext).GetGeometryName()
	if gtype == 'MULTILINESTRING':
		geom = ogr.CreateGeometryFromWkt(gtext)
		for tg in range(0, geom.GetGeometryCount()):
			gm = geom.GetGeometryRef(tg)
			gname = gm.GetGeometryName()
			x_p, y_p = extract_points(gm)
			x_list.append(x_p)
			y_list.append(y_p)
	elif gtype == 'LINESTRING':
		gm = ogr.CreateGeometryFromWkt(gtext)
		x_p, y_p = extract_points(gm)
		x_list.append(x_p)
		y_list.append(y_p)

	# x = ny.array(x_list)
	# y = ny.array(y_list)
	return x_list,y_list

def vector_details(file_path):
	try:
		with fiona.open(file_path, 'r') as source:
			fields = [(k,v) for k, v in source.schema['properties'].items()]
			geometry_type = source.schema['geometry']
			crs = fiona.crs.to_string(source.crs)
			bounds = source.bounds
			number_of_features = len(source)
		return fields, geometry_type, crs, bounds,number_of_features
	except Exception as ex:
		print("INFO: fiona read failure (likely not a vector file):", ex)
		return None

def raster_details(file_path):
	with rasterio.open(file_path) as dataset:
		bbox = dataset.bounds
		bounds = (bbox.left, bbox.bottom, bbox.right, bbox.top)
		number_of_cells = dataset.width * dataset.height
		if dataset.crs.is_valid:
			crs = dataset.crs.to_string()
		else:
			crs = 'invalid/unknown'
		bands = [(i, dtype) for i, dtype in zip(dataset.indexes, dataset.dtypes)]
	return bands, crs, bounds, number_of_cells

def add_columns_to_table(table_name, table_match, col_name_list,col_type_list, col_id, cursor, connection):
	for c in range(len(col_name_list)):
		sql_query = "alter table {0} add column {1} {2}".format(table_name,col_name_list[c],col_type_list[c])
		cursor.execute(sql_query)
		connection.commit()

		sql_query = "update %s set %s = 0"%(table_name,col_name_list[c])
		cursor.execute(sql_query)
		connection.commit()

		sql_query = '''
					update {0} set {1} = (select {2} from {3} as A where {4}.{5} = A.{6})
					'''.format(table_name,col_name_list[c],col_name_list[c],table_match,table_name,col_id,col_id)
		cursor.execute(sql_query)
		connection.commit()

		sql_query = "update %s set %s = 0 where %s is Null"%(table_name,col_name_list[c],col_name_list[c])
		cursor.execute(sql_query)
		connection.commit()


def add_columns_to_table_match_columns(table_name, table_match, col_name_list,col_type_list, col_id,col_id_match, cursor, connection):
	for c in range(len(col_name_list)):
		sql_query = "alter table {0} add column {1} {2}".format(table_name,col_name_list[c],col_type_list[c])
		cursor.execute(sql_query)
		connection.commit()

		sql_query = "update %s set %s = 0"%(table_name,col_name_list[c])
		cursor.execute(sql_query)
		connection.commit()

		sql_query = '''
					update {0} set {1} = (select {2} from {3} as A where {4}.{5} = A.{6})
					'''.format(table_name,col_name_list[c],col_name_list[c],table_match,table_name,col_id,col_id_match)
		cursor.execute(sql_query)
		connection.commit()

		sql_query = "update %s set %s = 0 where %s is Null"%(table_name,col_name_list[c],col_name_list[c])
		cursor.execute(sql_query)
		connection.commit()

def add_new_id_column(table_name,col_str,col_name,col_id_name, cursor, connection):
	sql_query = "select {0} from {1}".format(col_id_name,table_name)
	cursor.execute(sql_query)

	col_id_list = []
	read_layer = cursor.fetchall()
	for row in read_layer:
		col_id_list.append(row[0])


	sql_query = "alter table {0} add column {1} character varying".format(table_name,col_name)
	cursor.execute(sql_query)
	connection.commit()

	for c in col_id_list:
		cid = col_str + str(c)

		sql_query = '''
					update {0} set {1} = '{2}' where {3} = {4}
					'''.format(table_name,col_name,cid,col_id_name,c)
		cursor.execute(sql_query)
		connection.commit()

'''
Create the database connection
'''
conf = load_config()

try:
	conn = psycopg2.connect(**conf['database'])
except:
	print ("I am unable to connect to the database")


cur = conn.cursor()

# Give the input fields required for establishing the database connection
host = conf['database']['host']
dbname = conf['database']['database']
user = conf['database']['user']
password = conf['database']['password']


# Create the new database connection
conn2 = ogr.Open("PG: host='%s' dbname='%s' user='%s' password='%s'" %(host, dbname, user, password))
conn3 = ogr.Open("PG: host='%s' dbname='%s' user='%s' password='%s'" %(host, dbname, user, password))


'''
Step 1
Get the projection of the shapefile and write it to the database
With projection EPSG = 4326
'''

shp_file_path = os.path.join(conf['paths']['data'],'waters', 'inland_waters')

for file in os.listdir(shp_file_path):
	if file.endswith(".shp"):
		fpath = os.path.join(shp_file_path, file)
		flds, geom_type, cr, bds,n_f = vector_details(fpath)

		print (fpath)
		print (geom_type,cr)
		if geom_type.lower().strip() in ('point','multipoint'):
			fname = 'water_nodes'
		elif geom_type.lower().strip() in ('linestring','multilinestring'):
			fname = 'water_edges'
		else:
			fname = file.split(".")
			fname = fname[0]

		cr_split = cr.split(':')
		print (cr_split)
		fr_srid = [c for c in cr_split if c.isdigit() is True]
		if fr_srid:
			print (fr_srid)
		elif 'wgs84' in cr.lower().strip():
			fr_srid = '4326'

		command = "shp2pgsql -I -S -s {0}:4326 -d {1} {2} | psql -U {3} -d {4}".format(fr_srid,fpath,fname,conf['database']['user'], conf['database']['database'])
		# sp.call(command.split())
		sp.call(command, shell=True)

node_id = 'gid'
edge_id = 'gid'
node_layer = 'waterports'
edge_layer = 'waterroutes'


line_id_list = []
sql_query = '''SELECT {0} FROM {1}'''.format(edge_id,edge_layer)
cur.execute(sql_query)
read_layer = cur.fetchall()
for row in read_layer:
	line_id_list.append(row[0])

nid_lid_list = []

sql_query = '''SELECT A.{0} as nid,
			(select B.{1} from {2} as B order by st_distance(A.geom,B.geom) asc limit 1) as lid
			from {3} as A
			'''.format(node_id,edge_id,edge_layer,node_layer)
read_layer = conn2.ExecuteSQL(sql_query)
f = read_layer.GetNextFeature()
while f is not None:
	nid = f.GetField('nid')
	lid = f.GetField('lid')

	nid_lid_list.append((nid,lid))

	f.Destroy()
	f = read_layer.GetNextFeature()

'''
STEP 4:
We will create the new edge and node layers
'''
'''
create the edge layer
'''

sql_query = '''DROP TABLE IF EXISTS
			public.wateredges
			'''
sql_create = conn2.ExecuteSQL(sql_query)

sql_query = '''CREATE TABLE public.wateredges
			(
			edge_id character varying(254),
			node_f_id character varying(254),
			node_t_id character varying(254),
			gid integer,
			geom geometry(LineString,4326)
			)
			'''
sql_create = conn2.ExecuteSQL(sql_query)

'''
create the node layer
'''
sql_query = '''DROP TABLE IF EXISTS
			public.waternodes
			'''
sql_create = conn2.ExecuteSQL(sql_query)

sql_query = '''CREATE TABLE public.waternodes
			(
			node_id character varying(254),
			gid integer,
			geom geometry(Point,4326)
			)
			'''
sql_create = conn2.ExecuteSQL(sql_query)


u_lines = list(set(line_id_list))
dummy_pt = 20000
e_id = 0
network_list = []
for item in range(len(u_lines)):
	lc = u_lines[item]
	nl = [n for (n,l) in nid_lid_list if l == lc]
	if len(nl) > 0:
		nl = nl + [0]
		pt_tup_list = []
		sql_query = '''SELECT A.{0} AS nid,
					ST_AsText(ST_ClosestPoint(B.geom,A.geom)) AS pt_geom,
					ST_Line_Locate_Point(B.geom,ST_ClosestPoint(B.geom,A.geom)) as frac,
					ST_AsText(ST_StartPoint(B.geom)) as st_pt, ST_AsText(ST_EndPoint(B.geom)) as en_pt,
					ST_Distance(ST_ClosestPoint(B.geom,A.geom),ST_StartPoint(B.geom)) as st_pt_dist,
					ST_Distance(ST_ClosestPoint(B.geom,A.geom),ST_EndPoint(B.geom)) as en_pt_dist
					FROM {1} AS A,
					{2} AS B
					WHERE A.{3} IN {4}
					AND B.{5} = {6}
					'''.format(node_id,node_layer,edge_layer,node_id,str(tuple(nl)),edge_id,lc)
		read_layer = conn2.ExecuteSQL(sql_query)
		f = read_layer.GetNextFeature()
		while f is not None:
			nid = f.GetField('nid')

			pt_geom = f.GetField('pt_geom')
			frac = f.GetField('frac')
			st_pt = f.GetField('st_pt')
			en_pt = f.GetField('en_pt')
			st_pt_dist = f.GetField('st_pt_dist')
			en_pt_dist = f.GetField('en_pt_dist')

			pt_tup_list.append((nid,pt_geom,st_pt_dist,en_pt_dist,frac))

			f.Destroy()
			f = read_layer.GetNextFeature()

		'''
		ST_Line_Substring
		'''
		if len(pt_tup_list) > 0:
			pt_id_sorted = [p for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
			pt_geom_sorted = [w for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
			pt_dist_st_sorted = [x for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
			pt_dist_en_sorted = [y for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
			pt_frac_sorted = [z for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]

			if pt_dist_st_sorted[0] < 1e-10:
				pt_frac_sorted[0] = 0
				pt_geom_sorted[0] = st_pt

			if pt_dist_en_sorted[-1] < 1e-10:
				pt_frac_sorted[-1] = 1
				pt_geom_sorted[-1] = en_pt

			if min(pt_frac_sorted) > 0:
				pt_frac_sorted = [0] + pt_frac_sorted
				dummy_pt = dummy_pt + 1
				# pt_info = (dummy_pt,'No name','No type','No code','No name',elr,'No code')
				pt_id_sorted = [dummy_pt] + pt_id_sorted
				pt_geom_sorted = [st_pt] + pt_geom_sorted

			if max(pt_frac_sorted) < 1:
				pt_frac_sorted = pt_frac_sorted + [1]
				dummy_pt = dummy_pt + 1
				# pt_info = (dummy_pt,'No name','No type','No code','No name',elr,'No code')
				pt_id_sorted = pt_id_sorted + [dummy_pt]
				pt_geom_sorted = pt_geom_sorted + [en_pt]

			for p in range(len(pt_frac_sorted)-1):
				e_id = e_id + 1
				eid = 'wateredge_' + str(e_id)
				pt_st_frac = pt_frac_sorted[p]
				pt_en_frac = pt_frac_sorted[p+1]

				nf_id = pt_id_sorted[p]
				nt_id = pt_id_sorted[p+1]

				# print (pt_st_frac,pt_en_frac)
				nfid = 'waternode_' + str(nf_id)
				ntid = 'waternode_' + str(nt_id)

				sql_insert = '''INSERT INTO public.wateredges
							(edge_id,node_f_id,node_t_id,gid,geom)
							VALUES ('{0}','{1}','{2}',{3},
							ST_GeomFromText((SELECT ST_AsText(ST_Line_Substring(geom,{4},{5}))
							FROM {6} WHERE {7} = {8}),4326)
							)'''.format(eid,nfid,ntid,lc,pt_st_frac,pt_en_frac,edge_layer,edge_id,lc)
				create_layer = conn2.ExecuteSQL(sql_insert)


			for p in range(len(pt_id_sorted)):
				n_id = pt_id_sorted[p]
				nid = 'waternode_' + str(n_id)
				pt = pt_geom_sorted[p]
				sql_insert = '''INSERT INTO public.waternodes
							(node_id,gid,geom)
							VALUES ('{0}',{1},ST_GeomFromText('{2}',4326))
							'''.format(nid,n_id,pt)
				create_layer = conn2.ExecuteSQL(sql_insert)

	else:
		sql_query = '''SELECT ST_AsText(geom) AS l_geom,
					ST_AsText(ST_StartPoint(geom)) as st_pt,
					ST_AsText(ST_EndPoint(geom)) as en_pt
					FROM {0}
					WHERE gid = {1}
					'''.format(edge_layer,lc)
		read_layer = conn2.ExecuteSQL(sql_query)
		f = read_layer.GetNextFeature()
		while f is not None:
			gt = f.GetField('l_geom')
			st_pt = f.GetField('st_pt')
			en_pt = f.GetField('en_pt')

			# edge_id = edge_id + 1
			e_id += 1
			# network_list.append((e_id,nf_id,nt_id,lc,gt))

			eid = 'wateredge_' + str(e_id)

			dummy_pt = dummy_pt + 1
			nf_id = dummy_pt
			nfid = 'waternode_' + str(nf_id)

			dummy_pt = dummy_pt + 1
			nt_id = dummy_pt
			ntid = 'waternode_' + str(nt_id)

			sql_insert = '''INSERT INTO public.wateredges
						(edge_id,node_f_id,node_t_id,gid,geom)
						VALUES ('{0}','{1}','{2}',{3},
						ST_GeomFromText('{4}',4326))
						'''.format(eid,nfid,ntid,lc,gt)
			create_layer = conn2.ExecuteSQL(sql_insert)

			sql_insert = '''INSERT INTO public.waternodes
						(node_id,gid,geom)
						VALUES ('{0}',{1},
						ST_GeomFromText('{2}',4326))
						'''.format(nfid,nf_id,st_pt)
			create_layer = conn2.ExecuteSQL(sql_insert)

			sql_insert = '''INSERT INTO public.waternodes
						(node_id,gid,geom)
						VALUES ('{0}',{1},
						ST_GeomFromText('{2}',4326))
						'''.format(ntid,nt_id,en_pt)
			create_layer = conn2.ExecuteSQL(sql_insert)

			f.Destroy()
			f = read_layer.GetNextFeature()

	print ('done with line number %s with code %s'%(item,lc))


'''
STEP 6:
Remove the common nodes from the node and edge sets
If two nodes are within 10m of each other, they are considered the same node
'''

node_s_pairs = []
sql_query = '''SELECT A.node_id as a_n, B.node_id as b_n
			from waternodes as A, waternodes as B
			where ST_Distance(A.geom::geography,B.geom::geography) = 0
			and A.node_id != B.node_id
			'''
read_layer = conn2.ExecuteSQL(sql_query)
f = read_layer.GetNextFeature()
while f is not None:
	a_n = f.GetField('a_n')
	b_n = f.GetField('b_n')
	if ([a_n,b_n] not in node_s_pairs) and ([b_n,a_n] not in node_s_pairs):
		node_s_pairs.append([a_n,b_n])

	f.Destroy()
	f = read_layer.GetNextFeature()

# print len(node_s_pairs)
# print node_s_pairs[0:10]

'''
Get all the groups of common nodes
'''
'''
Sample list
l = [['a', 'b', 'c'], ['b', 'd', 'e'], ['k'], ['o', 'p'], ['e', 'f'], ['p', 'a'], ['d', 'g']]
'''
l = copy.deepcopy(node_s_pairs)
out = []
while len(l)>0:
	first, rest = l[0], l[1:]
	first = set(first)

	lf = -1
	while len(first)>lf:
		lf = len(first)

		rest2 = []
		for r in rest:
			if len(first.intersection(set(r)))>0:
				first |= set(r)
			else:
				rest2.append(r)
		rest = rest2

	out.append(first)
	l = rest

# print(len(out))
# print out[0:10]

'''
Test it!
'''
for i in out:
	nodes = sorted(list(i))
	del_nodes = nodes[1:] + ['0']
	sql_update = '''UPDATE wateredges SET node_f_id = '{0}'
				WHERE node_f_id IN {1}
				'''.format(nodes[0],str(tuple(nodes)))
	update_layer = conn2.ExecuteSQL(sql_update)

	sql_update = '''UPDATE wateredges SET node_t_id = '{0}'
				WHERE node_t_id IN {1}
				'''.format(nodes[0],str(tuple(nodes)))
	update_layer = conn2.ExecuteSQL(sql_update)

	sql_delete = '''DELETE FROM waternodes
				WHERE node_id IN {0}
				'''.format(str(tuple(del_nodes)))
	delete_layer = conn2.ExecuteSQL(sql_delete)



'''
STEP 7:
Get all the nodes with degree 1 and find if they are close to other edges
'''

cl_edges = []
nodes_edges = []

sql_query = '''SELECT A.node_id as a_n,
			(SELECT B.edge_id from wateredges as B where B.edge_id NOT IN (SELECT edge_id FROM wateredges WHERE node_f_id = A.node_id)
			and B.edge_id NOT IN (SELECT edge_id FROM wateredges WHERE node_t_id = A.node_id) order by st_distance(A.geom,B.geom) asc limit 1) as b_n,
			(SELECT ST_Distance(A.geom,B.geom)from wateredges as B where B.edge_id NOT IN (SELECT edge_id FROM wateredges WHERE node_f_id = A.node_id)
			and B.edge_id NOT IN (SELECT edge_id FROM wateredges WHERE node_t_id = A.node_id) order by st_distance(A.geom,B.geom) asc limit 1) as dist, A.geom
			FROM waternodes as A
			'''
read_layer = conn2.ExecuteSQL(sql_query)
f = read_layer.GetNextFeature()
while f is not None:
	a_n = f.GetField('a_n')
	b_n = f.GetField('b_n')
	d = f.GetField('dist')
	if d == 0:
		nodes_edges.append((a_n,b_n))
		if b_n not in cl_edges:
			cl_edges.append(b_n)

	f.Destroy()
	f = read_layer.GetNextFeature()

for lc in cl_edges:
	'''
	Find the nodes which have some edge matches
	'''
	nl = [n for (n,m) in nodes_edges if m == lc]
	if len(nl) > 0:
		nl = nl + ['0']
		pt_tup_list = []
		# print (nl,lc)
		sql_query = '''SELECT A.node_id AS n,
					ST_Line_Locate_Point(B.geom,ST_ClosestPoint(B.geom,A.geom)) as frac,
					B.node_f_id as st_pt, B.node_t_id as en_pt, B.gid as elr
					FROM public.waternodes AS A,
					public.wateredges AS B
					WHERE A.node_id IN {0}
					AND B.edge_id = '{1}'
					'''.format(str(tuple(nl)),lc)

		# print (sql_query)
		read_layer = conn2.ExecuteSQL(sql_query)
		f = read_layer.GetNextFeature()
		while f is not None:
			n = f.GetField('n')
			frac = f.GetField('frac')
			st_pt = f.GetField('st_pt')
			en_pt = f.GetField('en_pt')
			elr = f.GetField('elr')

			pt_tup_list.append((n,frac))

			f.Destroy()
			f = read_layer.GetNextFeature()

		# print pt_id_list
		# print pt_frac_list
		# print pt_dist_list
		'''
		ST_Line_Substring
		'''
		if len(pt_tup_list) > 0:
			pt_id_sorted = [x for (x,y) in sorted(pt_tup_list, key=lambda pair: pair[1])]
			pt_frac_sorted = [y for (x,y) in sorted(pt_tup_list, key=lambda pair: pair[1])]

			if min(pt_frac_sorted) > 0:
				pt_frac_sorted = [0] + pt_frac_sorted
				pt_id_sorted = [st_pt] + pt_id_sorted

			if max(pt_frac_sorted) < 1:
				pt_frac_sorted = pt_frac_sorted + [1]
				pt_id_sorted = pt_id_sorted + [en_pt]

			for p in range(len(pt_frac_sorted)-1):
				e_id = e_id + 1
				pt_st_frac = pt_frac_sorted[p]
				pt_en_frac = pt_frac_sorted[p+1]

				nf_id = pt_id_sorted[p]
				nt_id = pt_id_sorted[p+1]

				eid = 'wateredge_' + str(e_id)

				sql_insert = '''INSERT INTO public.wateredges
							(edge_id,node_f_id,node_t_id,gid,geom)
							VALUES ('{0}','{1}','{2}',{3},
							ST_GeomFromText((SELECT ST_AsText(ST_Line_Substring(geom,{4},{5}))
							FROM wateredges WHERE edge_id = '{6}'),4326)
							)'''.format(eid,nf_id,nt_id,elr,pt_st_frac,pt_en_frac,lc)
				create_layer = conn2.ExecuteSQL(sql_insert)



		sql_delete = '''DELETE FROM public.wateredges
					WHERE edge_id = '{0}'
					'''.format(lc)
		delete_layer = conn2.ExecuteSQL(sql_delete)


add_columns_to_table('wateredges', 'waterroutes', ['eid'], ['character varying'],'gid', cur, conn)
add_columns_to_table('waternodes', 'waterports', ['nid'], ['character varying'],'gid', cur, conn)

add_columns_to_table_match_columns('wateredges', 'waterwayedges', ['link','speed'], ['character varying','double precision'],'eid','edge_id', cur, conn)
add_columns_to_table_match_columns('waternodes', 'seaport_nodes', ['ten_cang','tinh'], ['character varying','character varying'],'nid','node_id', cur, conn)

# print (nid_lid_list)

# add_new_id_column('seaport_nodes','seanode_','node_id','gid',cur, conn)

'''
STEP 2:
Transform the multilinestring geometry to a linestring geometry
Create new railway edge table converting the multilinestrings to linestrings
'''

node_id = 'gid'
edge_id = 'gid'
node_layer = 'railway_nodes'
edge_layer = 'railway_edges'
node_attr = 'railwaylin'
edge_attr = 'railwaylin'

sql_query = '''DROP TABLE IF EXISTS
			public.railway_edges_linegeom
			'''
sql_create = conn2.ExecuteSQL(sql_query)

sql_query = '''CREATE TABLE public.railway_edges_linegeom
			(
			gid integer,
			linename character varying(254),
			geom geometry(LineString,4326)
			)
			'''
sql_create = conn2.ExecuteSQL(sql_query)


line_id_list = []
new_edge_id = 0
sql_query = '''SELECT {0}, ST_AsText(geom) FROM {1}'''.format(edge_attr,edge_layer)
cur.execute(sql_query)
read_layer = cur.fetchall()
for row in read_layer:
	link = row[0]
	gt = row[1]

	# print (gt)
	if gt is not None:
		g_x,g_y = get_geom_points(gt)

		# line_create = ogr.Geometry(ogr.wkbLineString)
		for j in range(0,len(g_x)):
			line_create = ogr.Geometry(ogr.wkbLineString)
			for i in range(0,len(g_x[j])):
				pt_x = g_x[j][i]
				pt_y = g_y[j][i]
				line_create.AddPoint_2D(pt_x,pt_y)

			line_gtext = line_create.ExportToWkt()
			new_edge_id += 1
			line_id_list.append(new_edge_id)
			sql_query = '''INSERT INTO railway_edges_linegeom (gid,linename,geom)
						VALUES ({0},'{1}',ST_GeomFromText('{2}',4326))
						'''.format(new_edge_id,link,str(line_gtext))

			cur.execute(sql_query)
			conn.commit()


'''
STEP 3:
Select the nodes and their matching edges
'''

edge_layer = 'railway_edges_linegeom'
edge_attr = 'linename'

nid_lid_list = []

sql_query = '''SELECT A.{0} as nid,
			(select B.{1} from {2} as B order by st_distance(A.geom,B.geom) asc limit 1) as lid,
			COALESCE((select B.{3} from {4} as B where A.{5} = B.{6} order by st_distance(A.geom,B.geom) asc limit 1),-9999) as sbid_lid,
			(select ST_distance(A.geom,B.geom) from {7} as B order by st_distance(A.geom,B.geom) asc limit 1) as cl_dist,
			COALESCE((select ST_distance(A.geom,B.geom) from {8} as B where A.{9} = B.{10} order by st_distance(A.geom,B.geom) asc limit 1),-9999) as sbid_dist
			from {11} as A
			'''.format(node_id,edge_id,edge_layer,edge_id,edge_layer,node_attr,edge_attr,edge_layer,edge_layer,node_attr,edge_attr,node_layer)
read_layer = conn2.ExecuteSQL(sql_query)
f = read_layer.GetNextFeature()
while f is not None:
	nid = f.GetField('nid')
	lid = f.GetField('lid')
	sbid_lid = f.GetField('sbid_lid')
	cl_dist = f.GetField('cl_dist')
	sbid_dist = f.GetField('sbid_dist')

	if lid != sbid_lid:
		if cl_dist > 100:
			match = 0
			if sbid_dist > 0:
				'''
				Match the station point to the line with the same business code
				'''
				nid_lid_list.append((nid,sbid_lid,match))
			else:
				'''
				Match the station point to the closest line
				'''
				nid_lid_list.append((nid,lid,match))
		else:
			match = 1
			if abs(sbid_dist - cl_dist) < 20:
				'''
				Match the station point to the line with the same business code
				'''
				nid_lid_list.append((nid,sbid_lid,match))
			else:
				'''
				Match the station point to the closest line
				'''
				nid_lid_list.append((nid,lid,match))
	else:
		match = 1
		'''
		Match the station point to the line with the same business code
		'''
		nid_lid_list.append((nid,sbid_lid,match))

	f.Destroy()
	f = read_layer.GetNextFeature()

# print (nid_lid_list)

'''
STEP 4:
We will create the new edge and node layers
'''
'''
create the edge layer
'''

sql_query = '''DROP TABLE IF EXISTS
			public.railnetworkedges
			'''
sql_create = conn2.ExecuteSQL(sql_query)

sql_query = '''CREATE TABLE public.railnetworkedges
			(
			edge_id character varying(254),
			node_f_id character varying(254),
			node_t_id character varying(254),
			gid integer,
			geom geometry(LineString,4326)
			)
			'''
sql_create = conn2.ExecuteSQL(sql_query)

'''
create the node layer
'''
sql_query = '''DROP TABLE IF EXISTS
			public.railnetworknodes
			'''
sql_create = conn2.ExecuteSQL(sql_query)

sql_query = '''CREATE TABLE public.railnetworknodes
			(
			node_id character varying(254),
			gid integer,
			geom geometry(Point,4326)
			)
			'''
sql_create = conn2.ExecuteSQL(sql_query)


'''
STEP 4:
Create the first iteration of the network nodes and egde sets
Based on the station points to rail line matches done in STEP 4
'''

u_lines = list(set(line_id_list))
dummy_pt = 20000
e_id = 0
network_list = []
for item in range(len(u_lines)):
	lc = u_lines[item]
	nlist = [(n,m) for (n,l,m) in nid_lid_list if l == lc]
	if len(nlist) > 0:
		'''
		Find the nodes which have some edge matches
		'''
		nl = [n for (n,m) in nlist if m == 1]
		if len(nl) > 0:
			nl = nl + [0]
			pt_tup_list = []
			sql_query = '''SELECT A.{0} AS nid,
						ST_AsText(ST_ClosestPoint(B.geom,A.geom)) AS pt_geom,
						ST_Line_Locate_Point(B.geom,ST_ClosestPoint(B.geom,A.geom)) as frac,
						ST_AsText(ST_StartPoint(B.geom)) as st_pt, ST_AsText(ST_EndPoint(B.geom)) as en_pt,
						ST_Distance(ST_ClosestPoint(B.geom,A.geom),ST_StartPoint(B.geom)) as st_pt_dist,
						ST_Distance(ST_ClosestPoint(B.geom,A.geom),ST_EndPoint(B.geom)) as en_pt_dist
						FROM {1} AS A,
						{2} AS B
						WHERE A.{3} IN {4}
						AND B.{5} = {6}
						'''.format(node_id,node_layer,edge_layer,node_id,str(tuple(nl)),edge_id,lc)
			read_layer = conn2.ExecuteSQL(sql_query)
			f = read_layer.GetNextFeature()
			while f is not None:
				nid = f.GetField('nid')

				pt_geom = f.GetField('pt_geom')
				frac = f.GetField('frac')
				st_pt = f.GetField('st_pt')
				en_pt = f.GetField('en_pt')
				st_pt_dist = f.GetField('st_pt_dist')
				en_pt_dist = f.GetField('en_pt_dist')

				pt_tup_list.append((nid,pt_geom,st_pt_dist,en_pt_dist,frac))

				f.Destroy()
				f = read_layer.GetNextFeature()

			'''
			ST_Line_Substring
			'''
			if len(pt_tup_list) > 0:
				pt_id_sorted = [p for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
				pt_geom_sorted = [w for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
				pt_dist_st_sorted = [x for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
				pt_dist_en_sorted = [y for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]
				pt_frac_sorted = [z for (p,w,x,y,z) in sorted(pt_tup_list, key=lambda pair: pair[-1])]

				if pt_dist_st_sorted[0] < 1e-10:
					pt_frac_sorted[0] = 0
					pt_geom_sorted[0] = st_pt

				if pt_dist_en_sorted[-1] < 1e-10:
					pt_frac_sorted[-1] = 1
					pt_geom_sorted[-1] = en_pt

				if min(pt_frac_sorted) > 0:
					pt_frac_sorted = [0] + pt_frac_sorted
					dummy_pt = dummy_pt + 1
					# pt_info = (dummy_pt,'No name','No type','No code','No name',elr,'No code')
					pt_id_sorted = [dummy_pt] + pt_id_sorted
					pt_geom_sorted = [st_pt] + pt_geom_sorted

				if max(pt_frac_sorted) < 1:
					pt_frac_sorted = pt_frac_sorted + [1]
					dummy_pt = dummy_pt + 1
					# pt_info = (dummy_pt,'No name','No type','No code','No name',elr,'No code')
					pt_id_sorted = pt_id_sorted + [dummy_pt]
					pt_geom_sorted = pt_geom_sorted + [en_pt]

				for p in range(len(pt_frac_sorted)-1):
					e_id = e_id + 1
					eid = 'railedge_' + str(e_id)
					pt_st_frac = pt_frac_sorted[p]
					pt_en_frac = pt_frac_sorted[p+1]

					nf_id = pt_id_sorted[p]
					nt_id = pt_id_sorted[p+1]

					# print (pt_st_frac,pt_en_frac)
					nfid = 'railnode_' + str(nf_id)
					ntid = 'railnode_' + str(nt_id)

					sql_insert = '''INSERT INTO public.railnetworkedges
								(edge_id,node_f_id,node_t_id,gid,geom)
								VALUES ('{0}','{1}','{2}',{3},
								ST_GeomFromText((SELECT ST_AsText(ST_Line_Substring(geom,{4},{5}))
								FROM {6} WHERE {7} = {8}),4326)
								)'''.format(eid,nfid,ntid,lc,pt_st_frac,pt_en_frac,edge_layer,edge_id,lc)
					create_layer = conn2.ExecuteSQL(sql_insert)

					# sql_insert = '''SELECT ST_AsText(ST_Line_Substring(geom,{0},{1})) as gtext
					# 			FROM {2} WHERE {3} = {4}
					# 			'''.format(pt_st_frac,pt_en_frac,edge_layer,edge_id,lc)
					# read_layer = conn2.ExecuteSQL(sql_insert)
					# f = read_layer.GetNextFeature()
					# while f is not None:
					# 	gt = f.GetField('gtext')

					# 	f.Destroy()
					# 	f = read_layer.GetNextFeature()

					# network_list.append((e_id,nf_id,nt_id,lc,gt))

				for p in range(len(pt_id_sorted)):
					n_id = pt_id_sorted[p]
					nid = 'railnode_' + str(n_id)
					pt = pt_geom_sorted[p]
					sql_insert = '''INSERT INTO public.railnetworknodes
								(node_id,gid,geom)
								VALUES ('{0}',{1},ST_GeomFromText('{2}',4326))
								'''.format(nid,n_id,pt)
					create_layer = conn2.ExecuteSQL(sql_insert)

	# 					# sql_insert = '''INSERT INTO public.railnetworknodes
	# 					# 			(node_id,name,type,fo_code,fo_name,elr_code,stn_code,geom)
	# 					# 			VALUES (%s,'%s','%s','%s','%s','%s','%s',
	# 					# 			ST_GeomFromText('%s',27700))
	# 					# 			'''%(n_id,n_a,t_y,f_c,f_n,e_l,s_t,pt)
	# 					# create_layer = conn2.ExecuteSQL(sql_insert)
	# 				else:
	# 					sql_insert = '''INSERT INTO public.railnetworknodes
	# 								(node_id,name,type,fo_code,fo_name,elr_code,stn_code,geom)
	# 								VALUES (%s,'No name','No type','No code','No name','%s','No code',
	# 								ST_GeomFromText('%s',27700))
	# 								'''%(n_id,elr,pt)
	# 					create_layer = conn2.ExecuteSQL(sql_insert)

		'''
		Find the nodes which have no edge matches
		'''
		# nl = [n for (n,m) in nlist if m == 0]
		# if len(nl) > 0:
		# 	for n in nl:
		# 		sql_insert = '''INSERT INTO public.railnetworknodes
		# 					(node_id,name,type,fo_code,fo_name,elr_code,stn_code,geom)
		# 					VALUES (%s,
		# 					(SELECT name
		# 					FROM public.stationspoint WHERE ogc_fid = %s),
		# 					(SELECT type
		# 					FROM public.stationspoint WHERE ogc_fid = %s),
		# 					(SELECT fo_code
		# 					FROM public.stationspoint WHERE ogc_fid = %s),
		# 					(SELECT fo_name
		# 					FROM public.stationspoint WHERE ogc_fid = %s),
		# 					(SELECT primary_el
		# 					FROM public.stationspoint WHERE ogc_fid = %s),
		# 					(SELECT stn_code
		# 					FROM public.stationspoint WHERE ogc_fid = %s),
		# 					ST_GeomFromText((SELECT ST_AsText(geom)
		# 					FROM public.stationspoint WHERE ogc_fid = %s),27700))
		# 					'''%(n,n,n,n,n,n,n,n)
		# 		create_layer = conn2.ExecuteSQL(sql_insert)

	else:
		sql_query = '''SELECT ST_AsText(geom) AS l_geom,
					ST_AsText(ST_StartPoint(geom)) as st_pt,
					ST_AsText(ST_EndPoint(geom)) as en_pt
					FROM {0}
					WHERE gid = {1}
					'''.format(edge_layer,lc)
		read_layer = conn2.ExecuteSQL(sql_query)
		f = read_layer.GetNextFeature()
		while f is not None:
			gt = f.GetField('l_geom')
			st_pt = f.GetField('st_pt')
			en_pt = f.GetField('en_pt')
			dummy_pt = dummy_pt + 1
			nf_id = dummy_pt
			dummy_pt = dummy_pt + 1
			nt_id = dummy_pt
			# edge_id = edge_id + 1
			e_id += 1
			# network_list.append((e_id,nf_id,nt_id,lc,gt))

			eid = 'railedge_' + str(e_id)
			nfid = 'railnode_' + str(nf_id)
			ntid = 'railnode_' + str(nt_id)

			sql_insert = '''INSERT INTO public.railnetworkedges
						(edge_id,node_f_id,node_t_id,gid,geom)
						VALUES ('{0}','{1}','{2}',{3},
						ST_GeomFromText('{4}',4326))
						'''.format(eid,nfid,ntid,lc,gt)
			create_layer = conn2.ExecuteSQL(sql_insert)

			sql_insert = '''INSERT INTO public.railnetworknodes
						(node_id,gid,geom)
						VALUES ('{0}',{1},
						ST_GeomFromText('{2}',4326))
						'''.format(nfid,nf_id,st_pt)
			create_layer = conn2.ExecuteSQL(sql_insert)

			sql_insert = '''INSERT INTO public.railnetworknodes
						(node_id,gid,geom)
						VALUES ('{0}',{1},
						ST_GeomFromText('{2}',4326))
						'''.format(ntid,nt_id,en_pt)
			create_layer = conn2.ExecuteSQL(sql_insert)

			f.Destroy()
			f = read_layer.GetNextFeature()

	print ('done with line number %s with code %s'%(item,lc))

# df = pd.DataFrame(network_list,columns = ['edge_id','node_f_id','node_t_id','line_code','geom'])
# df.to_csv('rail_network.csv',index = False)

'''
STEP 6:
Remove the common nodes from the node and edge sets
If two nodes are within 10m of each other, they are considered the same node
'''

node_s_pairs = []
sql_query = '''SELECT A.node_id as a_n, B.node_id as b_n
			from railnetworknodes as A, railnetworknodes as B
			where ST_Distance(A.geom::geography,B.geom::geography) = 0
			and A.node_id != B.node_id
			'''
read_layer = conn2.ExecuteSQL(sql_query)
f = read_layer.GetNextFeature()
while f is not None:
	a_n = f.GetField('a_n')
	b_n = f.GetField('b_n')
	if ([a_n,b_n] not in node_s_pairs) and ([b_n,a_n] not in node_s_pairs):
		node_s_pairs.append([a_n,b_n])

	f.Destroy()
	f = read_layer.GetNextFeature()

# print len(node_s_pairs)
# print node_s_pairs[0:10]

'''
Get all the groups of common nodes
'''
'''
Sample list
l = [['a', 'b', 'c'], ['b', 'd', 'e'], ['k'], ['o', 'p'], ['e', 'f'], ['p', 'a'], ['d', 'g']]
'''
l = copy.deepcopy(node_s_pairs)
out = []
while len(l)>0:
	first, rest = l[0], l[1:]
	first = set(first)

	lf = -1
	while len(first)>lf:
		lf = len(first)

		rest2 = []
		for r in rest:
			if len(first.intersection(set(r)))>0:
				first |= set(r)
			else:
				rest2.append(r)
		rest = rest2

	out.append(first)
	l = rest

# print(len(out))
# print out[0:10]

'''
Test it!
'''
for i in out:
	nodes = sorted(list(i))
	del_nodes = nodes[1:] + ['0']
	sql_update = '''UPDATE railnetworkedges SET node_f_id = '{0}'
				WHERE node_f_id IN {1}
				'''.format(nodes[0],str(tuple(nodes)))
	update_layer = conn2.ExecuteSQL(sql_update)

	sql_update = '''UPDATE railnetworkedges SET node_t_id = '{0}'
				WHERE node_t_id IN {1}
				'''.format(nodes[0],str(tuple(nodes)))
	update_layer = conn2.ExecuteSQL(sql_update)

	sql_delete = '''DELETE FROM railnetworknodes
				WHERE node_id IN {0}
				'''.format(str(tuple(del_nodes)))
	delete_layer = conn2.ExecuteSQL(sql_delete)



'''
STEP 7:
Get all the nodes with degree 1 and find if they are close to other edges
'''

cl_edges = []
nodes_edges = []

sql_query = '''SELECT A.node_id as a_n,
			(SELECT B.edge_id from railnetworkedges as B where B.edge_id NOT IN (SELECT edge_id FROM railnetworkedges WHERE node_f_id = A.node_id)
			and B.edge_id NOT IN (SELECT edge_id FROM railnetworkedges WHERE node_t_id = A.node_id) order by st_distance(A.geom,B.geom) asc limit 1) as b_n,
			(SELECT ST_Distance(A.geom,B.geom)from railnetworkedges as B where B.edge_id NOT IN (SELECT edge_id FROM railnetworkedges WHERE node_f_id = A.node_id)
			and B.edge_id NOT IN (SELECT edge_id FROM railnetworkedges WHERE node_t_id = A.node_id) order by st_distance(A.geom,B.geom) asc limit 1) as dist, A.geom
			FROM railnetworknodes as A
			'''
read_layer = conn2.ExecuteSQL(sql_query)
f = read_layer.GetNextFeature()
while f is not None:
	a_n = f.GetField('a_n')
	b_n = f.GetField('b_n')
	d = f.GetField('dist')
	if d == 0:
		nodes_edges.append((a_n,b_n))
		if b_n not in cl_edges:
			cl_edges.append(b_n)

	f.Destroy()
	f = read_layer.GetNextFeature()

for lc in cl_edges:
	'''
	Find the nodes which have some edge matches
	'''
	nl = [n for (n,m) in nodes_edges if m == lc]
	if len(nl) > 0:
		nl = nl + ['0']
		pt_tup_list = []
		# print (nl,lc)
		sql_query = '''SELECT A.node_id AS n,
					ST_Line_Locate_Point(B.geom,ST_ClosestPoint(B.geom,A.geom)) as frac,
					B.node_f_id as st_pt, B.node_t_id as en_pt, B.gid as elr
					FROM public.railnetworknodes AS A,
					public.railnetworkedges AS B
					WHERE A.node_id IN {0}
					AND B.edge_id = '{1}'
					'''.format(str(tuple(nl)),lc)

		# print (sql_query)
		read_layer = conn2.ExecuteSQL(sql_query)
		f = read_layer.GetNextFeature()
		while f is not None:
			n = f.GetField('n')
			frac = f.GetField('frac')
			st_pt = f.GetField('st_pt')
			en_pt = f.GetField('en_pt')
			elr = f.GetField('elr')

			pt_tup_list.append((n,frac))

			f.Destroy()
			f = read_layer.GetNextFeature()

		# print pt_id_list
		# print pt_frac_list
		# print pt_dist_list
		'''
		ST_Line_Substring
		'''
		if len(pt_tup_list) > 0:
			pt_id_sorted = [x for (x,y) in sorted(pt_tup_list, key=lambda pair: pair[1])]
			pt_frac_sorted = [y for (x,y) in sorted(pt_tup_list, key=lambda pair: pair[1])]

			if min(pt_frac_sorted) > 0:
				pt_frac_sorted = [0] + pt_frac_sorted
				pt_id_sorted = [st_pt] + pt_id_sorted

			if max(pt_frac_sorted) < 1:
				pt_frac_sorted = pt_frac_sorted + [1]
				pt_id_sorted = pt_id_sorted + [en_pt]

			for p in range(len(pt_frac_sorted)-1):
				e_id = e_id + 1
				pt_st_frac = pt_frac_sorted[p]
				pt_en_frac = pt_frac_sorted[p+1]

				nf_id = pt_id_sorted[p]
				nt_id = pt_id_sorted[p+1]

				eid = 'railedge_' + str(e_id)

				sql_insert = '''INSERT INTO public.railnetworkedges
							(edge_id,node_f_id,node_t_id,gid,geom)
							VALUES ('{0}','{1}','{2}',{3},
							ST_GeomFromText((SELECT ST_AsText(ST_Line_Substring(geom,{4},{5}))
							FROM railnetworkedges WHERE edge_id = '{6}'),4326)
							)'''.format(eid,nf_id,nt_id,elr,pt_st_frac,pt_en_frac,lc)
				create_layer = conn2.ExecuteSQL(sql_insert)



		sql_delete = '''DELETE FROM public.railnetworkedges
					WHERE edge_id = '{0}'
					'''.format(lc)
		delete_layer = conn2.ExecuteSQL(sql_delete)


add_columns_to_table('railnetworkedges', 'railway_edges_linegeom', ['linename'], ['character varying'],'gid', cur, conn)
add_columns_to_table('railnetworknodes', 'railway_nodes', ['railwaylin','name'], ['character varying','character varying'],'gid', cur, conn)
