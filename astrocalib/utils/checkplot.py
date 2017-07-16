#!/usr/bin/env python

from __future__ import print_function
from astropy import wcs
import aplpy
import matplotlib.pyplot as plt
from astropy.io import fits

def plot_check( img, w, x, y, ra, dec, ro = 2, rc = 3, origin = 0 ):

    hdu = fits.PrimaryHDU( data = img )
    fig = plt.figure( figsize = ( 8, 8 ) )
    f = aplpy.FITSFigure( hdu, figure = fig )
    f.show_grayscale()
    f.show_circles( x, y, ro, edgecolor = 'red' )
    f.show_circles( w.world2pix( ra, dec, origin = origin ), rc, edgecolor = 'green' )
    plt.show()
