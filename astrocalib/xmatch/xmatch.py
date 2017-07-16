#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from scipy.spatial import cKDTree
from astropy.table import hstack, vstack, Table
from astropy import wcs
from astropy.time import Time
import time
from math import sqrt


def crossmatch( X1, X2, max_distance = np.inf, check = False ):

    X1 = np.asarray( X1, dtype = float )
    X2 = np.asarray( X2, dtype = float )

    N1, D = X1.shape
    N2, D2 = X2.shape

    if D != D2:
        raise ValueError( 'Arrays must have the same second dimension' )

    kdt = cKDTree( X2 )

    dist, ind = kdt.query( X1, k=1, distance_upper_bound = max_distance )

    valid = np.where( dist < np.inf )

    if check:
        return( valid, ind[valid], dist[valid] )
    else:
        return( valid, ind[valid] )


def skybox( w, step = 10, extend = 1. ):

    xlist = range( 0, w._naxis[0], 100 ) + [w._naxis[0] - 1]
    ylist = range( 0, w._naxis[1], 100 ) + [w._naxis[1] - 1]
    edge = [ [0, p] for p in xlist ] + [ [ylist[-1], p] for p in xlist ] \
            + [ [p, 0] for p in ylist ] + [ [p, xlist[-1]] for p in ylist ]
    footprint = np.array( w.all_pix2world( edge, 0 ) )
    ra_range = footprint[:,0].min(), footprint[:,0].max()
    dec_range = footprint[:,1].min(), footprint[:,1].max()

    return ( ( ra_range[0] + ra_range[1] ) / 2,\
            ( dec_range[0] + dec_range[1] ) / 2,\
            ( ra_range[1] - ra_range[0] ) * extend, ( dec_range[1] - dec_range[0] ) * extend )


def catquery( box, cat = 'UCAC4', epoch = None, mag_range = ( np.NINF, np.inf ), countmax = 0, sortkey = 'default', sortrev = 0 ):


    if cat == 'UCAC4':
        try:
            from refcat import UCAC4
            cat = UCAC4()
        except:
            return( None )
        cat.extract( *box )
        stars = cat.data[ np.where( ( cat.data['mag1'] < mag_range[1] )\
                & ( cat.data['mag1'] > mag_range[0] ) ) ]

        if sortkey == 'default':
            sortkey = 'mag1'
        stars.sort( sortkey )
        if sortrev:
            stars = stars.reverse()

        if countmax:
            countmax = int( countmax )
            stars = stars[0:countmax]

        if epoch:
            stars['ra_obs'] = stars['ra'] + stars['pm_ra'] * ( epoch - stars['epoch_ra'] ) / 3600000.
            stars['dec_obs'] = stars['dec'] + stars['pm_dec'] * ( epoch - stars['epoch_dec'] ) / 3600000.
        else:
            stars['ra_obs'] = stars['ra']
            stars['dec_obs'] = stars['dec']

        return( stars )

    if cat == 'TGASPTYC':
        try:
            from refcat import TGASPTYC
            cat = TGASPTYC()
        except:
            return( None )
        cat.extract( *box )
        stars = cat.data[ np.where( ( cat.data['<Gmag>'] < mag_range[1] )\
                & ( cat.data['<Gmag>'] > mag_range[0] ) ) ]

        if sortkey == 'default':
            sortkey = '<Gmag>'
        stars.sort( sortkey )
        if sortrev:
            stars = stars.reverse()

        if countmax:
            countmax = int( countmax )
            stars = stars[0:countmax]

        if epoch:
            stars['ra_obs'] = stars['RAdeg'] + stars['pmRA'] * ( epoch - stars['Epoch'] ) / 3600000.
            stars['dec_obs'] = stars['DEdeg'] + stars['pmDE'] * ( epoch - stars['Epoch'] ) / 3600000.
        else:
            stars['ra_obs'] = stars['RAdeg']
            stars['dec_obs'] = stars['DEdeg']

        return( stars )

def matchcat( T, w, epoch=None, xkey = 'X_IMAGE', ykey = 'Y_IMAGE', cat = 'UCAC4', mag_range = ( np.NINF, np.inf ), refmax = 0, matchmax = 0, matchsortkey = None, refsortkey = 'default', matchsortrev = 0, refsortrev = 0, tol = 1. ):
    
    if isinstance( cat, Table ):
        stars = cat
    else:
        box = skybox( w )
        stars = catquery( box, cat = cat, mag_range = mag_range, epoch = epoch, countmax = refmax, sortkey = refsortkey,sortrev = refsortrev )
    rpix = w.all_world2pix( np.array( [stars['ra_obs'], stars['dec_obs']] ).T, 0 )
    tol = np.float64( tol )
    matchmax = np.int64( matchmax )
    if isinstance( tol, np.float64 ):
        mpix = np.array( [T[xkey], T[ykey]] ).T
        mind, rind = crossmatch( mpix, rpix, float( tol ) )
        match = hstack( [T[mind], stars[rind]] )
        ratio = len( match ) * 1. / len( T )
        if isinstance( matchmax, np.int64 ) and matchmax and matchsortkey:
            match.sort( matchsortkey )
            if matchsortrev:
                match.reverse()
            match = match[0:matchmax]
    elif isinstance( tol, np.ndarray ):
        if tol.ndim == 1:
            ratio = np.zeros( tol.shape )
            rparts, = tol.shape
            rstep = int( sqrt( w._naxis[0] ** 2 + w._naxis[1] ** 2 ) / ( 2 * rparts ) )
            match = Table()
            for i in range( rparts ):
                Tsub = T[np.where( np.int32( np.linalg.norm( np.array( [T[xkey] - w._naxis[0] / 2 + 0.5, T[ykey] - w._naxis[1] / 2 + 0.5] ), axis = 0 ) / rstep ) == i )]
                mpix = np.array( [Tsub[xkey], Tsub[ykey]] ).T
                mind, rind = crossmatch( mpix, rpix, tol[i] )
                sub = hstack( [Tsub[mind], stars[rind]] )
                ratio[i] = len( sub ) * 1. / len( Tsub )
                if isinstance( matchmax, np.int64 ) and matchmax and matchsortkey:
                    sub.sort( matchsortkey )
                    if matchsortrev:
                        sub.reverse()
                    sub = sub[0:matchmax]
                elif matchmax.shape == tol.shape and matchsortkey:
                    sub.sort( matchsortkey )
                    if matchsortrev:
                        sub.reverse()
                    sub = sub[0:matchmax[i]]
                match = vstack( [match, sub] )

        elif tol.ndim == 2:
            ratio = np.zeros( tol.shape )
            yparts, xparts = tol.shape
            ystep = int( w._naxis[1] / yparts )
            xstep = int( w._naxis[0] / xparts )
            match = Table()
            for i in range( yparts ):
                for j in range( xparts ):
                    Tsub = T[np.where( np.logical_and( np.int32( T[xkey] / xstep ) == j, np.int32( T[ykey] / ystep ) == i ) )]
                    mpix = np.array( [Tsub[xkey], Tsub[ykey]] ).T
                    mind, rind = crossmatch( mpix, rpix, tol[i,j] )
                    sub = hstack( [Tsub[mind], stars[rind]] )
                    ratio[i,j] = len( sub ) * 1. / len( Tsub )
                    if isinstance( matchmax, np.int64 ) and matchmax and matchsortkey:
                        sub.sort( matchsortkey )
                        if matchsortrev:
                            sub.reverse()
                        sub = sub[0:matchmax]
                    elif matchmax.shape == tol.shape and matchsortkey:
                        sub.sort( matchsortkey )
                        if matchsortrev:
                            sub.reverse()
                        sub = sub[0:matchmax[i,j]]
                    match = vstack( [match, sub] )

    return( match, ratio )
