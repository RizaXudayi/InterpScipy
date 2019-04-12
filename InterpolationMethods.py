# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:46:21 2019

@author: Riza

Description: this code investigates the properties of different Scipy interpolation
functions for a 2D unstructured field which is the solution of the Advection-Diffusion PDE.
Different methods are compared in terms of accuracy and speed. To see the results
simply run the code in a Python console.
For details on the interpolation functions, see the manual at:
    https://docs.scipy.org/doc/scipy/reference/interpolate.html

"""
import time
import numpy as np
import scipy.io as spio
from scipy import interpolate

from Domain import PolygonDomain2D
from ContourPlot import ContourPlot
from UtilityFunc import UF
uf = UF()

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#%% Data:

mat = spio.loadmat('coord.mat')
coord = mat['coord']                            # coordinates

# Exact concentration field:
mat = spio.loadmat('cEx.mat')
cEx = mat['cEx']
cEx = cEx.todense()

#%% Initialization:
    
# Domain:
vertices = np.array([[0.0, 0.0], [2.2, 0], [2.2, 2.2], [0.0, 2.2], 
                     [0.0, 1.1], [1.0, 1.1], [1.0, 1.0], [0.0, 1.0]])
domain = PolygonDomain2D(vertices=vertices)

# Contour plot class object:
contPlt = ContourPlot(domain)

# Dictionary for execution times:
constructTime = {}
executionTime = {}

#%% Radial basis function interpolation:

t = time.clock()
cEx_Rbf = interpolate.Rbf(coord[:,0], coord[:,1], cEx)
def cExRbf(x):
    """Function to compute the exact solution field."""
    return cEx_Rbf(x[:,0], x[:,1])[:, np.newaxis]
constructTime['Rbf'] = time.clock()-t

t = time.clock()
figNum=1
cRbf = contPlt.conPlot(cExRbf, figNum=figNum)
plt.title('radial basis functions')
plt.show()
executionTime['Rbf'] = time.clock()-t

#%% Grid data interpolation:

t = time.clock()
def cExGrid(x):
    """Function to compute the exact solution field."""
    return interpolate.griddata(coord, cEx, x, fill_value=0.0)
constructTime['griddata'] = time.clock()-t

t = time.clock()
figNum=2
cGriddata = contPlt.conPlot(cExGrid, figNum=figNum)
plt.title('grid data interpolator')
plt.show()
executionTime['griddata'] = time.clock()-t

#%% Linear ND interpolation:

t = time.clock()
cEx_lind = interpolate.LinearNDInterpolator(coord, cEx, fill_value=0.0)
def cExlind(x):
    """Function to compute the exact solution field."""
    return cEx_lind(x)
constructTime['lind'] = time.clock()-t

t = time.clock()
figNum=3
cLind = contPlt.conPlot(cExlind, figNum=figNum)
plt.title('linear ND interpolation')
plt.show()
executionTime['lind'] = time.clock()-t

#%% Nearest ND interpolation:

t = time.clock()
cEx_near = interpolate.NearestNDInterpolator(coord, cEx)
def cExNear(x):
    """Function to compute the exact solution field."""
    return cEx_near(x)
constructTime['near'] = time.clock()-t

t = time.clock()
figNum=4
cNear = contPlt.conPlot(cExNear, figNum=figNum)
plt.title('nearest ND interpolation')
plt.show()
executionTime['near'] = time.clock()-t

#%% Clough tocher 2D interpolation:

t = time.clock()
cEx_clough = interpolate.CloughTocher2DInterpolator(coord, cEx, fill_value=0.0)
def cExClough(x):
    """Function to compute the exact solution field."""
    return cEx_clough(x)
constructTime['clough'] = time.clock()-t

t = time.clock()
figNum=5
cClough = contPlt.conPlot(cExClough, figNum=figNum)
plt.title('clough tocher 2D interpolation')
plt.show()
executionTime['clough'] = time.clock()-t

#%% Results:

print('\n\nExecution times for different interpolation functions in Scipy:')

print('\tRadial basis function construction time: %2.3fsec' % constructTime['Rbf'])
print('\tRadial basis function interpolation time: %2.3fsec' % executionTime['Rbf'])
print('\tRadial basis function total time: %2.3fsec\n' % (constructTime['Rbf'] + executionTime['Rbf']) )

print('\tGrid data construction time: %2.3fsec' % constructTime['griddata'])
print('\tGrid data interpolation time: %2.3fsec' % executionTime['griddata'])
print('\tGrid data total time: %2.3fsec' % (constructTime['griddata'] + executionTime['griddata']) )
print('\tapproximation error compared to Rbf: %2.5f\n' % uf.l2Err(cRbf, cGriddata))

print('\tLinear ND interpolator construction time: %2.3fsec' % constructTime['lind'])
print('\tLinear ND interpolation time: %2.3fsec' % executionTime['lind'])
print('\tLinear ND interpolator total time: %2.3fsec' % (constructTime['lind'] + executionTime['lind']) )
print('\tapproximation error compared to Rbf: %2.5f\n' % uf.l2Err(cRbf, cLind))

print('\tNearest ND interpolator construction time: %2.3fsec' % constructTime['near'])
print('\tNearest ND interpolation time: %2.3fsec' % executionTime['near'])
print('\tNearest ND interpolator total time: %2.3fsec' % (constructTime['near'] + executionTime['near']) )
print('\tapproximation error compared to Rbf: %2.5f\n' % uf.l2Err(cRbf, cNear))

print('\tClough tocher 2D interpolator construction time: %2.3fsec' % constructTime['clough'])
print('\tClough tocher 2D interpolation time: %2.3fsec' % executionTime['clough'])
print('\tClough tocher 2D interpolator total time: %2.3fsec' % (constructTime['clough'] + executionTime['clough']) )
print('\tapproximation error compared to Rbf: %2.5f\n' % uf.l2Err(cRbf, cClough))

# Conclusion:
#   - most accurate (Rbf as basis): Clough, Linear = Grid, Nearest
#   - fastest interpolation time: Nearest, Linear, Clough, Grid, Rbf

#%%
#%% Scaling for large data:

print('\n#####################################################################################')
print(' A larger data case:')
largeTime = {}

# Large mesh:
mesh = domain.getMesh(discNum=150, bDiscNum=75)
coord2 = mesh.coordinates

print('\nRadial basis function interpolation ...')
t = time.clock()
cRbf2 = cExRbf(coord2)
largeTime['Rbf'] = time.clock()-t

print('\nGrid data interpolation ...')
t = time.clock()
cGriddata2 = cExGrid(coord2)
largeTime['griddata'] = time.clock()-t

print('\nLinear ND interpolation ...')
t = time.clock()
cLind2 = cExlind(coord2)
largeTime['lind'] = time.clock()-t

print('\nNearest ND interpolation ...')
t = time.clock()
cNear2 = cExNear(coord2)
largeTime['near'] = time.clock()-t

print('\nClough tocher 2D interpolation ...')
t = time.clock()
cClough2 = cExClough(coord2)
largeTime['clough'] = time.clock()-t

#%% Results:

print('\n\nExecution times for different interpolation functions in Scipy:')

print('\tRadial basis function interpolation time: %2.3fsec\n' % largeTime['Rbf'])

print('\tGrid data interpolation time: %2.3fsec' % largeTime['griddata'])
print('\tapproximation error compared to Rbf: %2.5f\n' % uf.l2Err(cRbf2, cGriddata2))

print('\tLinear ND interpolation time: %2.3fsec' % largeTime['lind'])
print('\tapproximation error compared to Rbf: %2.5f\n' % uf.l2Err(cRbf2, cLind2))

print('\tNearest ND interpolation time: %2.3fsec' % largeTime['near'])
print('\tapproximation error compared to Rbf: %2.5f\n' % uf.l2Err(cRbf2, cNear2))

print('\tClough tocher 2D interpolation time: %2.3fsec' % largeTime['clough'])
print('\tapproximation error compared to Rbf: %2.5f\n' % uf.l2Err(cRbf2, cClough2))

# Conclusion:
#   - most accurate (Rbf as basis): Clough, Linear = Grid, Nearest
#   - fastest interpolation time: Linear, Clough, Nearest, Grid, Rbf

#%%
#%% Extremely large data:

print('\n#####################################################################################')
print(' An extremely large data case (Rbf infeasible):')
exlargeTime = {}

# Large mesh:
mesh = domain.getMesh(discNum=1000, bDiscNum=75)
coord2 = mesh.coordinates

print('\nGrid data interpolation ...')
t = time.clock()
cGriddata2 = cExGrid(coord2)
exlargeTime['griddata'] = time.clock()-t

print('\nLinear ND interpolation ...')
t = time.clock()
cLind2 = cExlind(coord2)
exlargeTime['lind'] = time.clock()-t

print('\nNearest ND interpolation ...')
t = time.clock()
cNear2 = cExNear(coord2)
exlargeTime['near'] = time.clock()-t

print('\nClough tocher 2D interpolation ...')
t = time.clock()
cClough2 = cExClough(coord2)
exlargeTime['clough'] = time.clock()-t

#%% Results:

print('\n\nExecution times for different interpolation functions in Scipy:')

print('\tGrid data interpolation time: %2.3fsec' % exlargeTime['griddata'])
print('\tapproximation error compared to Clough: %2.5f\n' % uf.l2Err(cClough2, cGriddata2))

print('\tLinear ND interpolation time: %2.3fsec' % exlargeTime['lind'])
print('\tapproximation error compared to Clough: %2.5f\n' % uf.l2Err(cClough2, cLind2))

print('\tNearest ND interpolation time: %2.3fsec' % exlargeTime['near'])
print('\tapproximation error compared to Clough: %2.5f\n' % uf.l2Err(cClough2, cNear2))

print('\tClough tocher 2D interpolation time: %2.3fsec' % exlargeTime['clough'])

# Conclusion:
#   - most accurate (Clough as basis): Linear = Grid, Nearest
#   - fastest interpolation time: Linear, Grid, Clough, Nearest


#%% Final conclusions:

print('\n#####################################################################################')
print(' Final conclusions:')
print('- \'LinearNDInterpolator\' is faster and \'CloughTocher2DInterpolator\' is more accurate!')
print('- Their overal performance for this case study is similar and both are acceptable!')
print('#####################################################################################\n')













