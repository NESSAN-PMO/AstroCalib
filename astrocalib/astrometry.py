#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import ( absolute_import, print_function )
from astropy import log, wcs
from astropy.table import Table
from .xmatch import *
from .wfitting import *
import numpy as np

def initwcs( naxes, ra, dec, pixscale, rotate ):
    w = wcs.WCS()
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = np.array( naxes ) / 2 - 0.5
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [pixscale, pixscale]
    w.wcs.crota = [rotate, rotate]
    return( w )

def amcalib( objcat, w0, epoch = None, extend = 1.1, cat = 'TGASPTYC', mag_range = ( np.NINF, np.inf ), refmax = 0, refsortkey = 'default', xkey = 'X_IMAGE', ykey = 'Y_IMAGE', mkey = 'MAG_AUTO', tol = 10., matchmax = 0, wcstype = 'tan', order = 3, origin = 0 ):

    box = skybox( w0, extend = extend )
    if isinstance( cat, Table ):
        stars = cat
    else:
        stars = catquery( box, cat = cat, mag_range = mag_range, epoch = epoch, countmax = refmax, sortkey = refsortkey )

    d, r = matchcat( objcat, w0, xkey = xkey, ykey = ykey, cat = stars, tol = tol, matchsortkey = mkey, matchmax = matchmax )
    pc = np.array( [d[xkey], d[ykey]] ).T
    cc = np.array( [d['ra_obs'], d['dec_obs']] ).T
    w = wcsfitting( pc, cc, w0, wcstype = wcstype, order = order, origin = origin )
    return( w )
