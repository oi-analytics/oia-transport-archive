"""Road network flows
"""
import os
import sys

from collections import OrderedDict

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from shapely.geometry import LineString


from vtra.utils import *

def main():
	config = load_config()
	flows_file = os.path.join(config['paths']['data'], 'Results', 'Failure_shapefiles', 'weighted_edges_failures_national_road_2.shp')


	# plot_sets = [
	# 	{
	# 		'file_tag': 'reroute',
	# 		'no_access': [-1,0],
	# 		'legend_label': "(million USD/day)",
	# 		'divisor': 1000000,
	# 		'columns': ['min_tr_los','max_tr_los'],
	# 		'title_cols': ['Rerouting costs (min)','Rerouting costs (max)']
	# 	},
	# 	{
	# 		'file_tag': 'commodities',
	# 		'no_access':[-1,1],
	# 		'legend_label': "AADF (tons/day)",
	# 		'divisor': 1,
	# 		'columns': ['min_tons','max_tons'],
	# 		'title_cols': ['Total tonnage (min)','Total tonnage (max)']
	# 	},
	# 	{
	# 		'file_tag': 'economic',
	# 		'no_access':[-1,1],
	# 		'legend_label': "(million USD/day)",
	# 		'divisor': 1000000,
	# 		'columns': ['min_econ_l','max_econ_l'],
	# 		'title_cols': ['Economic losses (min)','Economic losses (max)']
	# 	},
	# 	{
	# 		'file_tag': 'total',
	# 		'no_access':[0,1],
	# 		'legend_label': "(million USD/day)",
	# 		'divisor': 1000000,
	# 		'columns': ['min_loss','max_loss'],
	# 		'title_cols': ['Total economic impact (min)','Total economic impact (max)']
	# 	},
	# 	{
	# 		'file_tag': 'adapt',
	# 		'no_access':[0,1],
	# 		'legend_label': "NPV (million USD)",
	# 		'divisor': 1000000,
	# 		'columns': ['min_adapt_','max_adapt_'],
	# 		'title_cols': ['NPV of adaptation over time (min)','NPV of adaptation over time (max)']
	# 	},
	# 	{
	# 		'file_tag': 'bcr',
	# 		'no_access':[0,1],
	# 		'legend_label': "BCR",
	# 		'divisor': 1,
	# 		'columns': ['min_bc_rat','max_bc_rat'],
	# 		'title_cols': ['BCR of adaptation over time (min)','BCR of adaptation over time (max)']
	# 	}
	# ]

	plot_sets = [
		{
			'file_tag': 'adapt',
			'no_access':[0,1],
			'legend_label': "NPV (million USD)",
			'divisor': 1000000,
			'columns': ['min_adapt_','max_adapt_'],
			'title_cols': ['NPV of adaptation over time (min)','NPV of adaptation over time (max)']
		},
		{
			'file_tag': 'bcr',
			'no_access':[0,1],
			'legend_label': "BCR",
			'divisor': 1,
			'columns': ['min_bc_rat','max_bc_rat'],
			'title_cols': ['BCR of adaptation over time (min)','BCR of adaptation over time (max)']
		}
	]

	for plot_set in plot_sets:
		for c in range(len(plot_set['columns'])):
			ax = get_axes()
			plot_basemap(ax, config['paths']['data'], highlight_region = [])
			scale_bar(ax, location=(0.8, 0.05))
			plot_basemap_labels(ax, config['paths']['data'])
			proj_lat_lon = ccrs.PlateCarree()

			# generate weight bins
			column = plot_set['columns'][c]
			if plot_set['file_tag'] == 'adapt':
				weights = [
					record.attributes[column]
					for record in shpreader.Reader(flows_file).records()
					if int(record.attributes['no_access']) in plot_set['no_access']
					and record.attributes[column] > 0 and record.attributes['max_loss'] > 0
				]

			elif plot_set['file_tag'] == 'bcr':
				weights = [
					record.attributes[column]
					for record in shpreader.Reader(flows_file).records()
					if int(record.attributes['no_access']) in plot_set['no_access']
					and record.attributes[column] > 1 and record.attributes['max_loss'] > 0
				]
			else:
				weights = [
					record.attributes[column]
					for record in shpreader.Reader(flows_file).records()
					if int(record.attributes['no_access']) in plot_set['no_access']
				]


			max_weight = max(weights)
			width_by_range = generate_weight_bins(weights)

			road_geoms_by_category = {
				'1': [],
				'2': [],
				'3': [],
				'4': [],
				'5': [],
				'6': []
			}

			if plot_set['file_tag'] == 'adapt':
				rec_set = [
					record
					for record in shpreader.Reader(flows_file).records()
					if int(record.attributes['no_access']) in plot_set['no_access']
					and record.attributes[column] > 0 and record.attributes['max_loss'] > 0
				]

			elif plot_set['file_tag'] == 'bcr':
				rec_set = [
					record
					for record in shpreader.Reader(flows_file).records()
					if int(record.attributes['no_access']) in plot_set['no_access']
					and record.attributes[column] > 1 and record.attributes['max_loss'] > 0
				]
			else:
				rec_set = [
					record
					for record in shpreader.Reader(flows_file).records()
					if int(record.attributes['no_access']) in plot_set['no_access']
				]

			for record in rec_set:

				cat = str(record.attributes['road_class'])
				if cat not in road_geoms_by_category:
					raise Exception
				geom = record.geometry

				val = record.attributes[column]

				buffered_geom = None
				for (nmin, nmax), width in width_by_range.items():
					if nmin <= val and val < nmax:
						buffered_geom = geom.buffer(width)

				if buffered_geom is not None:
					road_geoms_by_category[cat].append(buffered_geom)
				else:
					print("Feature was outside range to plot", record.attributes)

			styles = OrderedDict([
				('1',  Style(color='#000004', zindex=9, label='Class 1')), #red
				('2', Style(color='#2c115f', zindex=8, label='Class 2')), #orange
				('3', Style(color='#721f81', zindex=7, label='Class 3')), #blue
				('4',  Style(color='#b73779', zindex=6, label='Class 4')), #green
				('5', Style(color='#f1605d', zindex=5, label='Class 5')), #black
				('6', Style(color='#feb078', zindex=4, label='Class 6')), #grey
			])

			for cat, geoms in road_geoms_by_category.items():
				cat_style =styles[cat]
				ax.add_geometries(
					geoms,
					crs=proj_lat_lon,
					linewidth=0,
					facecolor=cat_style.color,
					edgecolor='none',
					zorder=cat_style.zindex
				)

			x_l = 102.3
			x_r = x_l + 0.4
			base_y = 14
			y_step = 0.4
			y_text_nudge = 0.1
			x_text_nudge = 0.1

			ax.text(
				x_l,
				base_y + y_step - y_text_nudge,
				plot_set['legend_label'],
				horizontalalignment='left',
				transform=proj_lat_lon,
				size=10)

			divisor = plot_set['divisor']
			for (i, ((nmin, nmax), width)) in enumerate(width_by_range.items()):
				y = base_y - (i*y_step)
				line = LineString([(x_l, y), (x_r, y)]).buffer(width)
				ax.add_geometries(
					[line],
					crs=proj_lat_lon,
					linewidth=0,
					edgecolor='#000000',
					facecolor='#000000',
					zorder=2)
				if nmin == max_weight:
					label = '>{:.2f}'.format(max_weight/divisor)
				else:
					label = '{:.2f}-{:.2f}'.format(nmin/divisor, nmax/divisor)
				ax.text(
					x_r + x_text_nudge,
					y - y_text_nudge,
					label,
					horizontalalignment='left',
					transform=proj_lat_lon,
					size=10)

			plt.title(plot_set['title_cols'][c], fontsize = 14)
			legend_from_style_spec(ax, styles)
			output_file = os.path.join(config['paths']['figures'], 'road_failure-map-{}-{}.png'.format(plot_set['file_tag'], column))
			save_fig(output_file)
			plt.close()



if __name__ == '__main__':
	main()
