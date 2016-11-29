#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:04:57 2016

@author: wolfensb
"""
from pyth_grib import GribFile, GribIndex, GribMessage
from pyth_grib.gribmessage import IndexNotSelectedError

filename = './lfsf00132500'
with GribFile(filename) as grib:
    for i in range(len(grib)):
        msg = GribMessage(grib)
        print(msg["shortName"])