#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 09:20:08 2019

@author: zhejunshen
"""

import helpy
import velocity
import ring_motion
import numpy as np
import matplotlib.pyplot as plt


class activity:
    def __init__(self,file_name, fps = 5.0, side = 25.25):
        self.prefix = file_name
        self.fps = fps
        self.side = side
        self.load_process()
        self.bulk_velocity()
        
    def load_process(self):
        self.x0, self.y0, self.R = ring_motion.boundary(self.prefix)
        data = helpy.load_data(self.prefix)
        data['o'] = (data['o'] + np.pi)%(2 * np.pi)
        max_frame = data['f'].max()
        tracksets = helpy.load_tracksets(data, run_track_orient=True, min_length = max_frame//2, run_repair = 'interp')
        track_prefix = {self.prefix: tracksets}
        v_data = velocity.compile_noise(track_prefix, width=(0.525,), cat = False, side = self.side, \
                                fps = self.fps, ring = True, x0= self.x0, y0 = self.y0, skip = 5, \
                                grad = True, start = 0)
        self.v_data = v_data[self.prefix]
        
    def bulk_velocity(self):
        v_p = np.empty(0)
        v_o = np.empty(0)
        v_t = np.empty(0)
        for key, value in self.v_data.items():
            mask = (value['r'] < self.R - self.side)
            v_p = np.append(v_p, value[mask]['v_p'])
            v_o = np.append(v_o, value[mask]['vomega'])
            v_t = np.append(v_t, value[mask]['v_t'])
        self.avg_vp = np.nanmean(v_p)
        self.avg_vt = np.nanmean(v_t)
        self.avg_vo = np.nanmean(v_o)
        self.std_vp = np.nanstd(v_p)
        self.std_vt = np.nanstd(v_t)
        self.std_vo = np.nanstd(v_o)
        
    def 