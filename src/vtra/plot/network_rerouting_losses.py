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

def main(mode):
	config = load_config()
	if mode == 'road':
		# flows_file = os.path.join(config['paths']['data'], 'Results', 'Failure_shapefiles', 'weighted_edges_failures_national_road_multi_modal_options.shp')
		flows_file = os.path.join(config['paths']['data'], 'Results', 'Failure_shapefiles', 'weighted_edges_failures_national_road_10_percent_shift.shp')
	elif mode == 'rail':
		flows_file = os.path.join(config['paths']['data'], 'Results', 'Failure_shapefiles', 'weighted_edges_failures_national_rail_multi_modal_options.shp')
	else:
		raise ValueError("Mode must be road or rail")

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
	# 		'file_tag': 'total',
	# 		'no_access':[0,1],
	# 		'legend_label': "(million USD/day)",
	# 		'divisor': 1000000,
	# 		'columns': ['min_loss','max_loss'],
	# 		'title_cols': ['Total economic impact (min)','Total economic impact (max)']
	# 	}
	# ]

	plot_sets = [
		{
			'file_tag': 'reroute',
			'no_access': [-1,0],
			'legend_label': "(million USD/day)",
			'divisor': 1000000,
			'columns': ['min_tr_los','max_tr_los'],
			'title_cols': ['Rerouting costs (min)','Rerouting costs (max)']
		},
	]

	for plot_set in plot_sets:
		for c in range(len(plot_set['columns'])):
			column = plot_set['columns'][c]
			print(" * Plotting", plot_set['file_tag'], column)

			ax = get_axes()
			plot_basemap(ax, config['paths']['data'], highlight_region = [])
			scale_bar(ax, location=(0.8, 0.05))
			plot_basemap_labels(ax, config['paths']['data'])
			proj_lat_lon = ccrs.PlateCarree()

			# generate weight bins
			# min_weight = round_sf(min(
			# 		min(
			# 			record.attributes[column]
			# 			for record
			# 			in stats[(sector, "current")]
			# 		)
			# 		for sector in sectors
			# ))
			# max_weight = round_sf(max(
			# 	max(
			# 		record.attributes[column]
			# 		for record
			# 		in stats[(sector, "current")]
			# 		)
			# 	for sector in sectors
			# ))
			# abs_max_weight = round_sf(max(
			# 	max(
			# 		abs(record.attributes[column])
			# 		for record
			# 		in stats[(sector, "current")]
			# 	)
			# 	for sector in sectors
			# ))

			weights = [
				record.attributes[column]
				for record in shpreader.Reader(flows_file).records()
				if int(record.attributes['no_access']) in plot_set['no_access']
			]

			# min_weight = round_sf(min(weights))
			# max_weight = round_sf(max(weights))
			# abs_max_weight = round_sf(max([abs(w) for w in weights]))

			min_weight = min(weights)
			max_weight = max(weights)
			abs_max_weight = max([abs(w) for w in weights])

			 # generate weight bins
			width_by_range = OrderedDict()
			colors_by_range = {}
			n_steps = 8

			# 10 colors - for each of n_steps
			# Colorbrewer http://colorbrewer2.org/#type=diverging&scheme=RdBu&n=8

			# positive_colors = [
			# 	'#fddbc7',
			# 	'#f4a582',
			# 	'#d6604d',
			# 	'#b2182b',
			# 	'#67001f',
			# ]
			# negative_colors = [
			# 	'#d1e5f0',
			# 	'#92c5de',
			# 	'#4393c3',
			# 	'#2166ac',
			# 	'#053061',
			# ]

			positive_colors = [
				'#f4a582',
				'#d6604d',
				'#b2182b',
				'#67001f',
			]
			negative_colors = [
				'#92c5de',
				'#4393c3',
				'#2166ac',
				'#053061',
			]
			width_step = 0.01

			mins = np.linspace(0, abs_max_weight, n_steps/2)

			maxs = list(mins)
			maxs.append(abs_max_weight*10)
			maxs = maxs[1:]

			assert len(maxs) == len(mins)

			# positive
			for i, (min_, max_) in reversed(list(enumerate(zip(mins, maxs)))):
				width_by_range[(min_, max_)] = (i + 2) * width_step
				colors_by_range[(min_, max_)] = positive_colors[i]

			# negative
			for i, (min_, max_) in enumerate(zip(mins, maxs)):
				width_by_range[(-max_, -min_)] = (i + 2) * width_step
				colors_by_range[(-max_, -min_)] = negative_colors[i]

			# width_by_range = generate_weight_bins(weights)

			# road_geoms_by_category = {
			# 	'1': [],
			# 	'2': [],
			# 	'3': [],
			# 	'4': [],
			# 	'5': [],
			# 	'6': []
			# }

			geoms_by_range = {}
			for value_range in width_by_range:
				geoms_by_range[value_range] = []

			for record in [rec for rec in shpreader.Reader(flows_file).records() if int(rec.attributes['no_access']) in plot_set['no_access']]:
				val = record.attributes[column]
				geom = record.geometry
				for nmin, nmax in geoms_by_range:
					if nmin <= val and val < nmax:
						geoms_by_range[(nmin, nmax)].append(geom)

			# plot
			for range_, width in width_by_range.items():
				ax.add_geometries(
					[geom.buffer(width) for geom in geoms_by_range[range_]],
					crs=proj_lat_lon,
					edgecolor='none',
					facecolor=colors_by_range[range_],
					zorder=2)

			# for record in [rec for rec in shpreader.Reader(flows_file).records() if int(rec.attributes['no_access']) in plot_set['no_access']]:
			# 	cat = str(record.attributes['road_class'])
			# 	if cat not in road_geoms_by_category:
			# 		raise Exception
			# 	geom = record.geometry

			# 	val = record.attributes[column]

			# 	buffered_geom = None
			# 	for (nmin, nmax), width in width_by_range.items():
			# 		if nmin <= val and val < nmax:
			# 			buffered_geom = geom.buffer(width)

			# 	if buffered_geom is not None:
			# 		road_geoms_by_category[cat].append(buffered_geom)
			# 	else:
			# 		print("Feature was outside range to plot", record.attributes)

			# styles = OrderedDict([
			# 	('1',  Style(color='#000004', zindex=9, label='Class 1')), #red
			# 	('2', Style(color='#2c115f', zindex=8, label='Class 2')), #orange
			# 	('3', Style(color='#721f81', zindex=7, label='Class 3')), #blue
			# 	('4',  Style(color='#b73779', zindex=6, label='Class 4')), #green
			# 	('5', Style(color='#f1605d', zindex=5, label='Class 5')), #black
			# 	('6', Style(color='#feb078', zindex=4, label='Class 6')), #grey
			# ])

			# for cat, geoms in road_geoms_by_category.items():
			# 	cat_style =styles[cat]
			# 	ax.add_geometries(
			# 		geoms,
			# 		crs=proj_lat_lon,
			# 		linewidth=0,
			# 		facecolor=cat_style.color,
			# 		edgecolor='none',
			# 		zorder=cat_style.zindex
			# 	)

			x_l = 102.3
			x_r = x_l + 0.4
			base_y = 14
			y_step = 0.4
			y_text_nudge = 0.1
			x_text_nudge = 0.1

			ax.text(
				x_l - x_text_nudge,
				base_y + y_step - y_text_nudge,
				plot_set['legend_label'],
				horizontalalignment='left',
				transform=proj_lat_lon,
				size=8)

			divisor = plot_set['divisor']

			i = 0
			for (nmin, nmax), width in width_by_range.items():
				if not geoms_by_range[(nmin, nmax)]:
					continue
				y = base_y - (i*y_step)
				i = i + 1
				line = LineString([(x_l, y), (x_r, y)])
				ax.add_geometries(
					[line.buffer(width)],
					crs=proj_lat_lon,
					linewidth=0,
					edgecolor=colors_by_range[(nmin, nmax)],
					facecolor=colors_by_range[(nmin, nmax)],
					zorder=2)
				if nmin == max_weight:
					label = '>{:.2f}'.format(max_weight/divisor)
				elif nmax == -abs_max_weight:
					label = '<{:.2f}'.format(-abs_max_weight/divisor)
				else:
					label = '{:.2f} to {:.2f}'.format(nmin/divisor, nmax/divisor)
				ax.text(
					x_r + x_text_nudge,
					y - y_text_nudge,
					label,
					horizontalalignment='left',
					transform=proj_lat_lon,
					size=8)

			plt.title(plot_set['title_cols'][c], fontsize = 14)
			# legend_from_style_spec(ax, styles)
			if mode == 'road':
				output_file = os.path.join(config['paths']['figures'], 'road_failure-map-{}-{}-multi-modal-options-10-shift.png'.format(plot_set['file_tag'], column))
			elif mode == 'rail':
				output_file = os.path.join(config['paths']['figures'], 'rail_failure-map-{}-{}-multi-modal-options.png'.format(plot_set['file_tag'], column))
			else:
				raise ValueError("Mode must be road or rail")
			save_fig(output_file)
			plt.close()
			print(" >", output_file)



if __name__ == '__main__':
	ok_values = ('road', 'rail')
	ok_values = ('road',)
	# if len(sys.argv) != 2 or sys.argv[1] not in ok_values:
	# 	print("Usage: ")
	# 	print("    {} <mode>".format(os.path.basename(__file__)))
	# 	print("Where mode is one of: {}. For example:".format(ok_values))
	# 	print("    python {} {}".format(__file__, ok_values[0]))
	# 	exit(-1)
	# main(sys.argv[1])
	for ok in ok_values:
		main(ok)
