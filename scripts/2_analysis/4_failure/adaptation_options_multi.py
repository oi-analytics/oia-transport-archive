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


def net_present_value(adaptation_options_dataframe,strategy_parameter,parameter_value_list,min_eal,max_eal,edge_options_dictionary,edge_width = 1.0, edge_length = 1.0):
	for param_val in parameter_value_list:
		
		st = adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'strategy'].values[0]
		st_min_benefit = edge_width*edge_length*adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'min_benefit'].sum() + min_eal
		st_max_benefit = edge_width*edge_length*adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'max_benefit'].sum() + max_eal

		min_npv = st_min_benefit - edge_width*edge_length*adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'max_cost'].sum() 
		max_npv = st_max_benefit - edge_width*edge_length*adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'min_cost'].sum()

		if adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'max_cost'].sum() > 0:
			min_bc_ratio = 1.0*st_min_benefit/(edge_width*edge_length*adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'max_cost'].sum())
		else:
			min_bc_ratio = 0

		if adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'min_cost'].sum() > 0:
			max_bc_ratio = 1.0*st_max_benefit/(edge_width*edge_length*adaptation_options_dataframe.loc[adaptation_options_dataframe[strategy_parameter] == param_val,'min_cost'].sum())
		else:
			max_bc_ratio = 0

		edge_options_dictionary.append({'strategy':st,'min_npv':min_npv,'max_npv':max_npv,'min_bc_ratio':min_bc_ratio,'max_bc_ratio':max_bc_ratio})

	return edge_options_dictionary

def main():
	data_path,calc_path,output_path = load_config()['paths']['data'],load_config()['paths']['calc'],load_config()['paths']['output']

	'''
	cols = ['band_name','band_num','climate_scenario','commune_id','commune_name','district_id','district_name',
			'edge_id','hazard_type','max_val','min_val','model','probability','province_id','province_name','sector',
			'year','length','road_cond','asset_type','width','min_econ_loss','max_econ_loss']
	'''

	start_year = 2016
	end_year = 2050
	truck_unit_wt = [5.0,20.0]

	discount_rate = 12.0
	total_discount_ratio = []
	for year in range(start_year,end_year):
		# total_discount_ratio += 1.0/math.pow(1.0 + 1.0*discount_rate/100.0,year - start_year)
		total_discount_ratio.append(1.0/math.pow(1.0 + 1.0*discount_rate/100.0,year - start_year))

	# total_discount_ratio = sum(total_discount_ratio_list)

	df_path =  os.path.join(data_path,'Adaptation_options','adaptation_options.xlsx')
	adaptation_df = pd.read_excel(df_path,sheet_name = 'adapt_options')

	adaptation_df['total_discount_ratio'] = sum(total_discount_ratio)

	min_maintain_discount_ratio_list = []
	max_maintain_discount_ratio_list = []

	for iter_, row in adaptation_df.iterrows():
		min_maintain_schedule = row['maintenance_times_min']
		max_maintain_schedule = row['maintenance_times_max']

		min_maintain_discount_ratio = 0
		max_maintain_discount_ratio = 0

		max_maintain_discount_years = np.arange(start_year,end_year,min_maintain_schedule)
		min_maintain_discount_years = np.arange(start_year,end_year,max_maintain_schedule)
		for year in max_maintain_discount_years[1:]:
			max_maintain_discount_ratio += 1.0/math.pow(1.0 + 1.0*discount_rate/100.0,year - start_year)

		for year in min_maintain_discount_years[1:]:
			min_maintain_discount_ratio += 1.0/math.pow(1.0 + 1.0*discount_rate/100.0,year - start_year)

		min_maintain_discount_ratio_list.append(min_maintain_discount_ratio)
		max_maintain_discount_ratio_list.append(max_maintain_discount_ratio)


	adaptation_df['min_maintain_discount_ratio'] = min_maintain_discount_ratio_list
	adaptation_df['max_maintain_discount_ratio'] = max_maintain_discount_ratio_list

	adaptation_df['min_benefit'] = adaptation_df['rehab_cost_min']*adaptation_df['total_discount_ratio']
	adaptation_df['max_benefit'] = adaptation_df['rehab_cost_max']*adaptation_df['total_discount_ratio']

	adaptation_df['min_cost'] = adaptation_df['adapt_cost_min']*adaptation_df['total_discount_ratio'] + adaptation_df['maintain_cost_min']*adaptation_df['min_maintain_discount_ratio']
	adaptation_df['max_cost'] = adaptation_df['adapt_cost_max']*adaptation_df['total_discount_ratio'] + adaptation_df['maintain_cost_max']*adaptation_df['max_maintain_discount_ratio']
	
	print (adaptation_df)

	'''
	Add new strategy
	cols = ['strategy','asset_type','asset_cond','location','height_m','adapt_cost_min','adapt_cost_max',
			'maintain_cost_min','maintain_cost_max','rehab_cost_min','rehab_cost_max','height_incr_min',
			'height_incr_max','maintenance_times_min','maintenance_times_max','cost_unit']
	'''
	cols = ['strategy','asset_type','asset_cond','location','height_m','cost_unit','min_cost','max_cost','min_benefit','max_benefit']
	adaptation_options = adaptation_df[cols]

	ad_opt = []
	st_desc = ['road change','roads','unpaved-paved','all',0,'$/m2']
	st_min_cost = adaptation_df.loc[adaptation_df['asset_cond'] == 'paved','min_cost'].sum()
	st_max_cost = adaptation_df.loc[adaptation_df['asset_cond'] == 'paved','max_cost'].sum()

	st_min_benefit = adaptation_df.loc[adaptation_df['asset_cond'] == 'unpaved','rehab_cost_min'].sum()*total_discount_ratio[0] + adaptation_df.loc[adaptation_df['asset_cond'] == 'paved','rehab_cost_min'].sum()*sum(total_discount_ratio[1:])
	st_max_benefit = adaptation_df.loc[adaptation_df['asset_cond'] == 'unpaved','rehab_cost_max'].sum()*total_discount_ratio[0] + adaptation_df.loc[adaptation_df['asset_cond'] == 'paved','rehab_cost_max'].sum()*sum(total_discount_ratio[1:])
	
	ad_opt.append(st_desc + [st_min_cost,st_max_cost,st_min_benefit,st_max_benefit])
	new_ht = 6
	st_desc = ['dyke building','dyke','rural','sea',new_ht,'million USD/km']
	st_min_cost = adaptation_df.loc[adaptation_df['height_m'] == 4,'min_cost'].sum() + (new_ht - 4)*adaptation_df.loc[adaptation_df['height_m'] == 4,'height_incr_min'].sum()*sum(total_discount_ratio)
	st_max_cost = adaptation_df.loc[adaptation_df['height_m'] == 4,'max_cost'].sum() + (new_ht - 4)*adaptation_df.loc[adaptation_df['height_m'] == 4,'height_incr_max'].sum()*sum(total_discount_ratio)

	st_min_benefit = adaptation_df.loc[adaptation_df['height_m'] == 4,'min_benefit'].sum()
	st_max_benefit = adaptation_df.loc[adaptation_df['height_m'] == 4,'max_benefit'].sum()
	ad_opt.append(st_desc + [st_min_cost,st_max_cost,st_min_benefit,st_max_benefit])

	ad_opt_df = pd.DataFrame(ad_opt,columns = cols)

	adaptation_options = adaptation_options.append(ad_opt_df, ignore_index=True)

	print (adaptation_options)

	cols = ['band_num','climate_scenario',
			'edge_id','hazard_type','max_val','min_val','model','probability',
			'year','exposure_length']

	# index_cols = ['edge_id','hazard_type','model','climate_scenario','year','road_cond','asset_type','width','road_length']
	selection_criteria = ['commune_id','hazard_type','model','climate_scenario','year']
	filter_cols = ['edge_id','exposed_length'] + selection_criteria

	# provinces to consider 
	province_list = ['Lao Cai','Binh Dinh','Thanh Hoa']
	province_terrian = ['mountain','flat','flat']
	growth_scenarios = [(5,'low'),(6.5,'forecast'),(10,'high')]
	base_year = 2016
	types = ['min','max']

	fail_scenarios_data = os.path.join(output_path,'hazard_scenarios','province_roads_hazard_intersections.xlsx')
	rd_prop_file = os.path.join(data_path,'Roads','road_properties','road_properties.xlsx')

	duration_max = [10,15,20,25,30]
	length_thr = 100.0

	for prn in range(len(province_list)):
	# for prn in range(1,3):
		province = province_list[prn]
		# set all paths for all input files we are going to use
		province_name = province.replace(' ','').lower()

		all_edge_fail_scenarios = pd.read_excel(fail_scenarios_data,sheet_name = province_name)
		all_edge_fail_scenarios.loc[all_edge_fail_scenarios['probability'] == 'none', 'probability'] = 1.0
		all_edge_fail_scenarios['probability'] = pd.to_numeric(all_edge_fail_scenarios['probability'])
		all_edge_fail_scenarios.rename(columns={'length': 'exposure_length'}, inplace=True)
		all_edge_fail_scenarios = all_edge_fail_scenarios[cols]
		all_edge_fail_scenarios = all_edge_fail_scenarios.drop_duplicates(subset=cols, keep=False)
				

		all_edges = list(set(all_edge_fail_scenarios['edge_id'].values.tolist()))
		all_edge_fail_scenarios['road_cond'] = 'unknown'
		all_edge_fail_scenarios['asset_type'] = 'unknown'
		all_edge_fail_scenarios['width'] = 0
		all_edge_fail_scenarios['road_length'] = 0

		edges_in = os.path.join(data_path,'Roads','{}_roads'.format(province_name),'vietbando_{}_edges.shp'.format(province_name))
		edges = province_shapefile_to_dataframe(edges_in,province_terrian[prn],rd_prop_file)
		edge_attr = list(zip(edges['edge_id'].values.tolist(),edges['road_cond'].values.tolist(),edges['asset_type'].values.tolist(),edges['width'].values.tolist(),edges['length'].values.tolist()))
		# print (edge_attr)
		
		edge_attr = [e for e in edge_attr if e[0] in all_edges]
		for e in edge_attr:
			all_edge_fail_scenarios.loc[all_edge_fail_scenarios['edge_id'] == e[0], 'road_cond'] = e[1]
			all_edge_fail_scenarios.loc[all_edge_fail_scenarios['edge_id'] == e[0], 'asset_type'] = e[2]
			all_edge_fail_scenarios.loc[all_edge_fail_scenarios['edge_id'] == e[0], 'width'] = e[3]
			all_edge_fail_scenarios.loc[all_edge_fail_scenarios['edge_id'] == e[0], 'road_length'] = 1000.0*e[4]

		# all_edge_fail_scenarios['percent_exposure'] = 100.0*all_edge_fail_scenarios['exposure_length']/all_edge_fail_scenarios['road_length']
		# df_path = os.path.join(output_path,'hazard_scenarios','roads_hazard_intersections_{}.csv'.format(province_name))
		# all_edge_fail_scenarios.to_csv(df_path,index = False)
		
		# all_edge_fail_scenarios = all_edge_fail_scenarios.set_index(index_cols)
		# scenarios = list(set(all_edge_fail_scenarios.index.values.tolist()))

		all_edge_fail_scenarios = all_edge_fail_scenarios.set_index(selection_criteria)
		scenarios = list(set(all_edge_fail_scenarios.index.values.tolist()))
		
		multiple_ef_list = []
		for criteria in criteria_set:
			if len(all_edge_fail_scenarios.loc[criteria,'edge_id'].index) == 1:
				efail = [all_edge_fail_scenarios.loc[criteria,'edge_id'].item()]
			else:
				efail = list(set(all_edge_fail_scenarios.loc[criteria,'edge_id'].values.tolist()))

			flength = all_edge_fail_scenarios.loc[criteria,'length'].sum()
			pflength = flength/all_edge_fail_scenarios.loc[criteria,'road_length'].sum()

			criteria_dict = {**(dict(list(zip(selection_criteria,criteria)))),**{'exposed_length':flength}}
			multiple_ef_list.append((efail,criteria_dict))

		for tr_wt in truck_unit_wt: 
			flow_output_excel = os.path.join(output_path,'failure_results','multiple_edge_failures_totals_{0}_{1}_tons_projections.xlsx'.format(province_name,int(tr_wt)))
			adapt_output_excel = os.path.join(output_path,'failure_results','multiple_edge_failures_scenarios_{0}_{1}_tons_adapt_options.xlsx'.format(province_name,int(tr_wt)))
			excl_wrtr = pd.ExcelWriter(adapt_output_excel)
			for grth in growth_scenarios:
				loss_time_series = pd.read_excel(flow_output_excel,sheet_name = grth[1])
				for t in range(len(types)):
					loss_cols = []
					for year in range(start_year,end_year):
						# col = '{0}_econ_loss_{1}'.format(types[t],year)
						loss_cols.append('{0}_econ_loss_{1}'.format(types[t],year))

					# print (cols)
					# loss_time_series = loss_time_series[['edge_id']+cols]
					# print (loss_time_series)
					loss_time_series['{}_total_loss'.format(types[t])] = loss_time_series[loss_cols].sum(axis = 1)
				
				loss_time_series = loss_time_series[['edge_id','min_econ_loss_{}'.format(base_year),'max_econ_loss_{}'.format(base_year),'min_total_loss','max_total_loss']]
				loss_time_series = loss_time_series[loss_time_series['max_total_loss'] > 0]

				print ('done with loss estimation')
				for dur in range(len(duration_max)):
					scenarios_list = []
					for sc in scenarios:
						edge = sc[0]
						road_cond = sc[-4]
						width = sc[-2]
						# print (road_cond,width)

						sub_df = all_edge_fail_scenarios.loc[sc]
						if len(sub_df.index) == 1:
							min_exposure_len = sub_df.loc[sc,'exposure_length']
							max_exposure_len = sub_df.loc[sc,'exposure_length']
							min_height = sub_df.loc[sc,'min_val']
							max_height = sub_df.loc[sc,'max_val']
							min_band_num = sub_df.loc[sc,'band_num']
							max_band_num = sub_df.loc[sc,'band_num']
							min_per = sub_df.loc[sc,'percent_exposure']
							max_per = sub_df.loc[sc,'percent_exposure']
							min_dur = 0.01*duration_max[dur]*sub_df.loc[sc,'percent_exposure']
							if sub_df.loc[sc,'exposure_length'] < length_thr:
								max_dur = 0.01*duration_max[dur]*sub_df.loc[sc,'percent_exposure']
								risk_wt = 0.01*duration_max[dur]*sub_df.loc[sc,'percent_exposure']*sub_df.loc[sc,'probability']
							else:
								max_dur = duration_max[dur]
								risk_wt = duration_max[dur]*sub_df.loc[sc,'probability']


						else:
							prob = list(set(sub_df['probability'].values.tolist()))
							min_height = max(sub_df['min_val'].values.tolist())
							max_height = max(sub_df['max_val'].values.tolist())
							min_band_num = min(sub_df['band_num'].values.tolist())
							max_band_num = max(sub_df['band_num'].values.tolist())

							sub_df = sub_df.set_index('probability')
							exposure_len = []
							per = []
							risk_wt = 0
							for pr in prob:
								# per_exp = sub_df.loc[pr,'percent_exposure'].sum()
								if sub_df.loc[pr,'percent_exposure'].sum() > 100.0:
									exposure_len.append(100.0*sub_df.loc[pr,'exposure_length'].sum()/sub_df.loc[pr,'percent_exposure'].sum())
									per.append(100.0)
									risk_wt += 1.0*duration_max[dur]*pr
								else:
									exposure_len.append(sub_df.loc[pr,'exposure_length'].sum())
									if sub_df.loc[pr,'exposure_length'].sum() < length_thr:
										per.append(sub_df.loc[pr,'percent_exposure'].sum())
										risk_wt += 0.01*duration_max[dur]*sub_df.loc[pr,'percent_exposure'].sum()*pr
									else:
										per.append(100.0)
										risk_wt += duration_max[dur]*pr

				
							max_exposure_len = max(exposure_len)
							min_exposure_len = min(exposure_len)

							min_per = min(per)
							max_per = max(per)
							min_dur = 0.01*duration_max[dur]*min_per
							max_dur = 0.01*duration_max[dur]*max_per


						sc_list = list(sc) + [min_band_num,max_band_num,min_height,max_height,min_per,max_per,min_dur,max_dur,min_exposure_len,max_exposure_len]
						# scenarios_list.append(list(sc) + [min_band_num,max_band_num,min_height,max_height,min_per,max_per,min_dur,max_dur,min_exposure_len,max_exposure_len,risk_wt])

						if edge in loss_time_series['edge_id'].values.tolist():
							min_daily_loss = loss_time_series.loc[loss_time_series['edge_id'] == edge,'min_econ_loss_{}'.format(base_year)].sum()
							max_daily_loss = loss_time_series.loc[loss_time_series['edge_id'] == edge,'max_econ_loss_{}'.format(base_year)].sum()

							edge_options = []
							min_edge_ead = sum(total_discount_ratio)*risk_wt*loss_time_series.loc[loss_time_series['edge_id'] == edge,'min_total_loss'].sum()
							max_edge_ead = sum(total_discount_ratio)*risk_wt*loss_time_series.loc[loss_time_series['edge_id'] == edge,'max_total_loss'].sum()
							if max_height > 4:
								max_height = 6

							if max_height > 0:
								edge_options = net_present_value(adaptation_options,'height_m',[max_height],min_edge_ead,max_edge_ead,edge_options,edge_width = 1.0, edge_length = 1000.0*max_exposure_len)
				
							if road_cond == 'unpaved':
								edge_options = net_present_value(adaptation_options,'asset_cond',['unpaved','unpaved-paved'],min_edge_ead,max_edge_ead,edge_options,edge_width = width, edge_length = max_exposure_len)
							
							if road_cond == 'paved':
								edge_options = net_present_value(adaptation_options,'asset_cond',['paved'],min_edge_ead,max_edge_ead,edge_options,edge_width = width, edge_length = max_exposure_len)

						else:
							min_daily_loss = 0
							max_daily_loss = 0
							min_edge_ead = 0
							max_edge_ead = 0
							edge_options = [{'strategy':'none','min_npv':0,'max_npv':0,'min_bc_ratio':0,'max_bc_ratio':0}]

						for options in edge_options:
							scenarios_list.append(sc_list + [min_daily_loss,max_daily_loss,min_edge_ead,max_edge_ead,options['strategy'],options['min_npv'],options['max_npv'],options['min_bc_ratio'],options['max_bc_ratio']])



					new_cols = ['min_band','max_band','min_height','max_height','min_exposure_percent','max_exposure_percent',
								'min_duration','max_duration','min_exposure_length','max_exposure_length','min_daily_loss','max_daily_loss',
								'min_npv_nooption','max_npv_nooption','adapt_strategy','min_npv','max_npv','min_bc_ratio','max_bc_ratio']
					scenarios_df = pd.DataFrame(scenarios_list,columns = index_cols + new_cols)
					# df_path = os.path.join(output_path,'hazard_scenarios','roads_hazard_intersections_{}_risks.csv'.format(province_name))
					# scenarios_df.to_csv(df_path,index = False)
					scenarios_df.to_excel(excl_wrtr,grth[1] + '_' + str(duration_max[dur]),index = False)
					excl_wrtr.save()

					print ('Done with {0} in {1} tons {2} growth scenario {3} days'.format(province_name,tr_wt,grth[1],duration_max[dur]))


if __name__ == "__main__":
	main()