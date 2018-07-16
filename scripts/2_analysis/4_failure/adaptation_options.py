# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 12:00:00 2018
@author: Raghav Pant
"""

import geopandas as gpd
import pandas as pd
import os
import json
import igraph as ig
import numpy as np
import math


def net_present_value(adaptation_cost,maintenance_cost,rehabilitation_cost,economic_loss,time_horizon,discount_ratio):
	npv = 0
	for i in range(0,time_horizon):
		npv += ((rehabilitation_cost[i] + economic_loss[i]) - (0adaptation_cost[i] + maintenance_cost[i]))/(math.power(1 + discount_ratio,i))

	return npv