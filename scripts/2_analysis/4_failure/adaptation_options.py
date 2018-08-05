# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:00:00 2018
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
import sys
import math
import copy 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from scripts.utils import load_config
from scripts.transport_network_creation import province_shapefile_to_network, add_igraph_generalised_costs_province_roads, province_shapefile_to_dataframe


def net_present_value(adaptation_cost,maintenance_cost,rehabilitation_cost,economic_loss,time_horizon,discount_ratio):
	npv = 0
	for i in range(0,time_horizon):
		npv += ((rehabilitation_cost[i] + economic_loss[i]) - (adaptation_cost[i] + maintenance_cost[i]))/(math.power(1 + discount_ratio,i))

	return npv

def main():
	data_path,calc_path,output_path = load_config()['paths']['data'],load_config()['paths']['calc'],load_config()['paths']['output']

	# provinces to consider 
	province_list = ['Lao Cai','Binh Dinh','Thanh Hoa']
	province_terrian = ['mountain','flat','flat']

	fail_scenarios_data = os.path.join(output_path,'hazard_scenarios','province_roads_hazard_intersections.xlsx')
	rd_prop_file = os.path.join(data_path,'Roads','road_properties','road_properties.xlsx')

	'''
	Path OD flow disruptions
	'''
	# for prn in range(len(province_list)):
	for prn in range(0,1):
		province_ods_df = []
		province = province_list[prn]
		# set all paths for all input files we are going to use
		province_name = province.replace(' ','').lower()


		df_path = os.path.join(output_path,'hazard_scenarios','roads_hazard_intersections_{}.csv'.format(province_name))
		all_edge_fail_scenarios = pd.read_csv(df_path)
		



if __name__ == "__main__":
	main()