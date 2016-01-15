import os
import numpy as np
from numpy import *
import subprocess
import binvox_rw
import fftw3f

class Shape:
    '''
    The Shape class include the properties and functions to store 3D raterized
    (voxel) model and perform operations on it
    '''
    
    def __init__(self):
        '''
        Initializes some parameters for the voxel model
        '''
        # The voxel data (to be stored as 3D numpy array of 0's and 1's
        self.voxel = array([])
        # Fourier transform of the voxel data
        self.voxel_ft = array([])
        self.visible = True
        # the actual resolution of the shape taking scale into consideration
        self.size = 64
        # resolution input by the user
        self.resolution = 64
        self.scale = 1
        self.filename = ""

    def read_voxel(self):
        '''
        Reads in a triangulated 3D model file (.obj, .stl, etc.), rasterizes 
        it using 'binvox', and saves the data as 3D numpy array of 0's and 1's
        '''
        if len(self.filename) != 0:
            binvox = self.filename[:self.filename.rfind('.')] + '.binvox'
            
            if not os.path.isfile(binvox):
                subprocess.call("./binvox -d "+ str(self.size) +
                                    " " + self.filename, shell = True)
            
            fid = open(binvox, 'r')
            model = binvox_rw.read_as_3d_array(fid)
            
            if model.dims[0] != self.size:
                os.remove(binvox)
                subprocess.call("./binvox -d "+ str(self.size) +
                                    " " + self.filename, shell = True)
                fid = open(binvox, 'r')
                model = binvox_rw.read_as_3d_array(fid)
            
            self.voxel = 1*model.data
            if self.scale != 1:
                self.pad_voxel([self.resolution] * 3)

    def write_voxel(self, filename):
        '''
        Write the voxel model data into a .binvox file
        '''
        if len(filename) != 0 and not self.isempty():
            fp = open(filename, 'w')
            data = self.voxel > 0
            dims = list(self.get_voxel_shape())
            translate = [0.0, 0.0, 0.0]
            scale = 1.0
            axis_order = 'xyz'
            model = binvox_rw.Voxels(data, dims, translate, scale, axis_order)
            binvox_rw.write(model, fp)
    
    def set_voxel(self, voxel):
        '''
        Sets the voxel field to voxel and updates the resolution
        '''
        self.voxel = voxel
        self.set_resolution(self.get_voxel_shape()[0])

    def set_voxel_ft(self, voxel_ft):
        '''
        Sets the voxel_ft field to voxel_ft
        '''
        self.voxel_ft = voxel_ft

    def set_size(self):
        '''
        Computes and sets the size at which the shape has to be rasterized
        Takes into consideration scale and resolution defined by the user
        '''
        self.size = max(1, int(self.scale * self.resolution))

    def set_resolution(self, resolution):
        '''
        Sets the resolution and updates the size field
        '''
        self.resolution = resolution
        self.set_size()

    def set_scale(self, scale):
        '''
        Sets the scale and updates the size field
        '''
        self.scale = scale
        self.set_size()

    def set_filename(self, filename):
        self.filename = filename

    def set_visibility(self, flag = True):
        self.visible = flag

    def toggle_visibility(self):
        self.visible = not self.visible

    def get_voxel(self):
        return self.voxel

    def get_voxel_ft(self):
        return self.voxel_ft

    def get_size(self):
        return self.size

    def get_scale(self):
        return self.scale

    def get_filename(self):
        return self.filename

    def get_voxel_shape(self):
        return array(self.voxel.shape)

    def isempty(self):
        '''
        Returns if the voxel field is empty
        '''
        return self.voxel.size == 0

    def get_sublevel_set(self, level):
        '''
        Returns the sublevel set of the voxel model at level
        The output is a 3D numpy array with 1's in all cell were voxel field
        has value larger than 99.99% of level and 0's elsewhere
        99.99% is used to account for any precision error
        '''
        return 1 * (self.voxel > 0.9999*level)

    def get_volume(self):
        '''
        Returns the volume or the number of high cells in the voxel model
        '''
        sublevel_set = self.get_sublevel_set(1)
        return np.count_nonzero(sublevel_set)

    def pad_voxel(self, dims):
        '''
        Pads the voxel field with zero to enflate the size upto 'dims'
        The actual data is centered in the 3D array of size 'dims'
        '''
        sz = self.get_voxel_shape()
        voxel = self.voxel
        self.voxel = zeros(dims, dtype = 'f')

        sid = (dims - sz)/2
        eid = sid + sz
        self.voxel[sid[0]:eid[0], sid[1]:eid[1], sid[2]:eid[2]] = voxel

    def fourier_transform(self):
        '''
        Computes the Fast Fourier Transform of the rasterized 3D model
        '''
        voxel = self.voxel.astype('f')
        self.voxel_ft = voxel.astype('F')
        trans = fftw3f.Plan(voxel, self.voxel_ft, direction='forward')
        trans()

    def inverse_fourier_transform(self):
        '''
        Computes Inverse Fourier Transform to get back the rasterized 3D model
        '''
        if self.voxel_ft.size == 0:
            pass
        else:
            self.voxel = zeros(self.voxel_ft.shape, dtype = 'f')
            trans = fftw3f.Plan(self.voxel_ft, self.voxel, direction='backward')
            trans()

    def normalize(self):
        '''
        Normalizes the Inverse Fourier Tranform
        '''
        self.voxel /= prod(self.voxel.shape)
        self.voxel = fft.fftshift(self.voxel)

    def display(self, mlab, contours = [1], color = (1, 0, 0), opacity = 0.5):
        '''
        Displays the voxel model if the visible flag is set to true and the
        voxel field is not empty
        '''
        if self.visible and not self.isempty():
            mlab.contour3d(self.voxel, contours = contours,
                color = color, opacity = opacity)
            
