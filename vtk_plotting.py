import vtk
from vtk.util import numpy_support
import h5py
import numpy as np
import pyvista as pv
from pyvista import examples

basepath, epoch, realfake, AorB = None, 1, 'fake', 'B'
f = h5py.File(basepath+str(epoch)+'_'+realfake+'_'+AorB+'.vox', 'r').get('data').value

f = f[0,0,:,:,:]

# binary thresholding of intensity range [-1 1]
threshold = 0.
f[f <= threshold] = -1.
f[f > threshold] = 1.

# plot 3d segmentation mask with pyvista
mesh = pv.wrap(f)
plotter = pv.Plotter()    
plotter.add_mesh_threshold(mesh, cmap='PuBuGn', smooth_shading=True, lighting=True)
cpos = plotter.show() 