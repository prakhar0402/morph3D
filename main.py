import numpy as np
from numpy import *
import os
import subprocess
import binvox_rw
import fftw3f
from mayavi import mlab

class shape:

	def __init__(self):
		self.voxel = array([])
		self.voxel_ft = array([])

	def read_voxel(self, filename):
		fid = open(filename, 'r')
		model = binvox_rw.read_as_3d_array(fid)
		self.voxel = model.data

	def set_voxel(self, voxel):
		self.voxel = voxel

	def set_voxel_ft(self, voxel_ft):
		self.voxel_ft = voxel_ft

	def get_voxel(self):
		return self.voxel

	def get_voxel_ft(self):
		return self.voxel_ft

	def get_voxel_size(self):
		return array(self.voxel.shape)

	def get_sublevel_set(self, level):
		return 1 * (self.voxel > 0.9999*level)

	def get_volume(self):
		sublevel_set = self.get_sublevel_set(1)
		return np.count_nonzero(sublevel_set)

	def pad_voxel(self, dims):
		sz = self.get_voxel_size()
		voxel = self.voxel
		self.voxel = zeros(dims, dtype = 'f')

		sid = (dims - sz)/2
		eid = sid + sz
		self.voxel[sid[0]:eid[0], sid[1]:eid[1], sid[2]:eid[2]] = voxel

	def fourier_transform(self):
		voxel = self.voxel.astype('f')
		self.voxel_ft = voxel.astype('F')
		trans = fftw3f.Plan(voxel, self.voxel_ft, direction = 'forward')
		trans()

	def inverse_fourier_transform(self):
		if self.voxel_ft.size == 0:
			pass
		else:
			self.voxel = zeros(self.voxel_ft.shape, dtype = 'f')
			trans = fftw3f.Plan(self.voxel_ft, self.voxel, direction = 'backward')
			trans()

	def normalize(self):
		self.voxel /= prod(self.voxel.shape)
		self.voxel = fft.fftshift(self.voxel)

	def display(self, contours = [1], color = (1, 0, 0), opacity = 0.5):
		sf = mlab.pipeline.scalar_field(self.voxel)
		mlab.pipeline.iso_surface(sf, contours = contours, color = color, opacity = opacity)

def get_norm_corr(alpha, beta):
	alpha.fourier_transform()
	beta.fourier_transform()
	
	corr = shape()
	corr.set_voxel_ft(alpha.get_voxel_ft() * beta.get_voxel_ft())
	
	corr.inverse_fourier_transform()
	corr.normalize()
	
	return corr

def minkowski_sum(alpha, beta):
	corr = get_norm_corr(alpha, beta)
	level = 0.0001*(beta.get_volume()-0.5)

	msum = shape()
	msum = msum.set_voxel(corr.get_sublevel_set(level))

	return msum

def minkowski_diff(alpha, beta):
	corr = get_norm_corr(alpha, beta)
	level = 1*(beta.get_volume()-0.5)

	mdiff = shape()
	mdiff = mdiff.set_voxel(corr.get_sublevel_set(level))

	return mdiff

def minkowski_sum_and_diff(alpha, beta):
	corr = get_norm_corr(alpha, beta)
	level_sum = 0.0001*(beta.get_volume()-0.5)
	level_diff = 1*(beta.get_volume()-0.5)

	msum = shape()
	mdiff = shape()

	msum.set_voxel(corr.get_sublevel_set(level_sum))
	mdiff.set_voxel(corr.get_sublevel_set(level_diff))

	return msum, mdiff

# Input file names
input_files = ['data/testpart0.obj', 'data/testpart1.obj']

# Binvox file names for the input files
binvox0 = input_files[0][:input_files[0].rfind('.')] + '.binvox'
binvox1 = input_files[1][:input_files[1].rfind('.')] + '.binvox'

# Writing binvox files if not already exist
if not os.path.isfile(binvox0):
    subprocess.call('./binvox -d 64 ' + input_files[0], shell = True)
if not os.path.isfile(binvox1):
    subprocess.call('./binvox -d 64 ' + input_files[1], shell = True)

# Initializing
alpha = shape()
beta = shape()

# Reading the binvox files into shapes
alpha.read_voxel(binvox0)
beta.read_voxel(binvox1)

# Getting the total size for convolution
sza = alpha.get_voxel_size()
szb = beta.get_voxel_size()
szc = sza + szb
dims = [int(pow(2, ceil(log(dim)/log(2)))) for dim in szc]

# Padding the shapes with zeros
alpha.pad_voxel(dims)
beta.pad_voxel(dims)

msum, mdiff = minkowski_sum_and_diff(beta, alpha)
print msum.get_volume(), mdiff.get_volume()

alpha.display(contours = [1], color = (1, 0, 0), opacity = 0.5)
beta.display(contours = [1], color = (0, 0, 1), opacity = 1.0)
msum.display(contours = [1], color = (1, 1, 0), opacity = 0.3)
mdiff.display(contours = [1], color = (0, 1, 0), opacity = 0.3)

mlab.show()

