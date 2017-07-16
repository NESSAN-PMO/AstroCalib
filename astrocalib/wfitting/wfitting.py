#!/usr/bin/env python

from __future__ import print_function
from astropy import wcs
from astropy.coordinates import SkyCoord
import scipy.optimize as optimize
import numpy as np
from math import sqrt

def wcsbuilder( wcstype = 'tan', order = 0 ):
    w = wcs.WCS()
    if wcstype == 'tan':
        p = np.zeros( 10 )
        w.wcs.ctype = ["RA---TAN","DEC--TAN"]
    elif wcstype == 'tan-sip':
        p = np.zeros( 4 + ( order + 1 ) * ( order + 2 ) )
        w.wcs.ctype = ["RA---TAN-SIP","DEC--TAN-SIP"]
    else:
        return( None )
    def func( p ):
        w.wcs.crpix = [p[0], p[1]]
        w.wcs.crval = [p[2], p[3]]
        w.wcs.cdelt = [p[4], p[5]]
        w.wcs.pc = [[p[6], p[7]], [p[8], p[9]]]
        if wcstype == 'tan-sip':
            a = np.zeros( ( order + 1, order + 1 ) )
            b = np.zeros( ( order + 1, order + 1 ) )
            ap = np.zeros( a.shape )
            bp = np.zeros( b.shape )
            n = 0
            for i in range( order + 1 ):
                for j in range( order + 1 ):
                    if i + j <= 1 or i + j > order:
                        continue
                    a[i,j] = p[n + 10]
                    b[i,j] = p[n + len( p ) / 2 + 5]
                    n += 1
            w.sip = wcs.Sip( a, b, ap, bp, w.wcs.crpix )
        return( w )
    return( p, func )

def residuals( p, pc, cc, builder, origin = 0 ):
    w = builder( p )
    rd_p = w.all_pix2world( pc, origin )
    sc_p = SkyCoord( rd_p[:,0], rd_p[:,1], unit = 'deg' )
    sc_c = SkyCoord( cc[:,0], cc[:,1], unit = 'deg' )
    sep = sc_p.separation( sc_c )
    return( sep )

def wcsfitting( pc, cc, w0 = None, wcstype = 'tan', order = 3, origin = 0 ):

    p_init, builder = wcsbuilder( wcstype = wcstype, order = order )
    if not w0:
        return
    p_init[0:2] = w0.wcs.crpix
    p_init[2:4] = w0.wcs.crval
    p_init[4:6] = wcs.utils.proj_plane_pixel_scales( w0 )
    p_init[6:10] = ( w0.pixel_scale_matrix / p_init[4:6] ).reshape( ( 4, ) )
    p = optimize.leastsq( residuals, p_init, args = ( pc, cc, builder, origin ), ftol = 0.0001, full_output = True )
    return( builder( p[0] ) )

