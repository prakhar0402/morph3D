# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
# By default, the PySide binding will be used. If you want the PyQt bindings
# to be used, you need to set the QT_API environment variable to 'pyqt'
#os.environ['QT_API'] = 'pyqt'

# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
from pyface.qt import QtGui, QtCore
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#   import sip
#   sip.setapi('QString', 2)

from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor

import numpy as np
from numpy import *
import subprocess
import binvox_rw
import fftw3f

################################################################################
# The Shape class
class Shape:

    def __init__(self):
        self.voxel = array([])
        self.voxel_ft = array([])
        self.visible = True
        self.size = 64
        self.filename = ""

    def read_voxel(self):
        if len(self.filename) != 0:
            binvox = self.filename[:self.filename.rfind('.')] + '.binvox'
            
            if not os.path.isfile(binvox):
                subprocess.call("./binvox -d "+ str(self.size) + " " + self.filename, shell = True)
            
            fid = open(binvox, 'r')
            model = binvox_rw.read_as_3d_array(fid)
            
            if model.dims[0] != self.size:
                os.remove(binvox)
                subprocess.call("./binvox -d "+ str(self.size) + " " + self.filename, shell = True)
                fid = open(binvox, 'r')
                model = binvox_rw.read_as_3d_array(fid)
            
            self.voxel = 1*model.data

    def set_voxel(self, voxel):
        self.voxel = voxel
        self.set_size(self.get_voxel_shape()[0])

    def set_voxel_ft(self, voxel_ft):
        self.voxel_ft = voxel_ft
        self.set_size(self.get_voxel_shape()[0])

    def set_size(self, size):
        self.size = size

    def set_filename(self, filename):
        self.filename = filename

    def set_visible(self, flag = True):
        self.visible = flag

    def toggle_visibility(self):
        self.visible = not self.visible

    def get_voxel(self):
        return self.voxel

    def get_voxel_ft(self):
        return self.voxel_ft

    def get_size(self):
        return self.size

    def get_filename(self):
        return self.filename

    def get_voxel_shape(self):
        return array(self.voxel.shape)

    def isempty(self):
        return self.voxel.size == 0

    def get_sublevel_set(self, level):
        return 1 * (self.voxel > 0.9999*level)

    def get_volume(self):
        sublevel_set = self.get_sublevel_set(1)
        return np.count_nonzero(sublevel_set)

    def pad_voxel(self, dims):
        sz = self.get_voxel_shape()
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

    def display(self, mlab, contours = [1], color = (1, 0, 0), opacity = 0.5):
        if self.visible and not self.isempty():
            mlab.contour3d(self.voxel, contours = contours, color = color, opacity = opacity)


################################################################################



################################################################################
# The Visualization class
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self):
        self.scene.mlab.clf()
        alpha.display(self.scene.mlab, contours = [1], color = (1, 0, 0), opacity = 0.5)
        beta.display(self.scene.mlab, contours = [1], color = (0, 0, 1), opacity = 1.0)
        msum.display(self.scene.mlab, contours = [1], color = (1, 1, 0), opacity = 0.3)
        mdiff.display(self.scene.mlab, contours = [1], color = (0, 1, 0), opacity = 0.3)

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False), resizable=True)

################################################################################

################################################################################
# The MayaviQWidget class
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

################################################################################

################################################################################
# The Window class
class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Morph3D")
        self.home()
        
    def home(self):
        self.container = QtGui.QWidget()
        self.layout = QtGui.QGridLayout(self.container)
        
        self.file_grid = QtGui.QGridLayout()
        
        self.morph = QtGui.QPushButton("COMPUTE", self)
        self.morph.clicked.connect(self.compute)
        self.morph.resize(self.morph.sizeHint())
        self.file_grid.addWidget(self.morph, 0, 0)
        
        self.part0 = QtGui.QPushButton("Click to select the first part.", self)
        self.part0.clicked.connect(self.file_open0)
        self.part0.resize(self.part0.sizeHint())
        self.file_grid.addWidget(self.part0, 1, 0)
        
        self.part1 = QtGui.QPushButton("Click to select the second part.", self)
        self.part1.clicked.connect(self.file_open1)
        self.part1.resize(self.part1.sizeHint())
        self.file_grid.addWidget(self.part1, 2, 0)
        
        self.res_label = QtGui.QLabel()
        self.res_label.setText("Resolution")
        self.file_grid.addWidget(self.res_label, 0, 1)
        
        self.res0 = QtGui.QSpinBox()
        self.res0.valueChanged[int].connect(self.on_change0)
        self.file_grid.addWidget(self.res0, 1, 1)
        
        self.res1 = QtGui.QSpinBox()
        self.res1.valueChanged[int].connect(self.on_change1)
        self.file_grid.addWidget(self.res1, 2, 1)
        
        self.layout.addLayout(self.file_grid, 0, 0)
        
        label = QtGui.QLabel(self.container)
        label.setText("Your QWidget at (%d, %d)" % (1, 1))
        label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)
        self.layout.addWidget(label, 1, 1)
        
        self.mayavi_widget = MayaviQWidget(self.container)
        self.layout.addWidget(self.mayavi_widget, 1, 0)
        
        self.container.show()
        self.setCentralWidget(self.container)
        
    def file_open0(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        self.part0.setText(name)
        alpha.set_filename(name)
        alpha.read_voxel()
        self.update()
        
    def file_open1(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        self.part1.setText(name)
        beta.set_filename(name)
        beta.read_voxel()
        self.update()
        
    def on_change0(self, value):
        alpha.set_size(value)
        
    def on_change1(self, value):
        beta.set_size(value)
        
    def compute(self):
        compute()
        self.morph.setText('Done')
        self.update()
    
    def update(self):
        self.mayavi_widget.visualization.update_plot()
        
    def reset(self):
        global msum, mdiff
        self.morph.setText("COMPUTE")
        msum = shape()
        mdiff = shape()

################################################################################
# Global functions

def get_norm_corr(alpha, beta):
    alpha.fourier_transform()
    beta.fourier_transform()

    corr = Shape()
    corr.set_voxel_ft(alpha.get_voxel_ft() * beta.get_voxel_ft())

    corr.inverse_fourier_transform()
    corr.normalize()

    return corr

def minkowski_sum(alpha, beta):
    global msum
    corr = get_norm_corr(alpha, beta)
    level = 0.0001*(beta.get_volume()-0.5)

    msum = Shape()
    msum = msum.set_voxel(corr.get_sublevel_set(level))

    return msum

def minkowski_diff(alpha, beta):
    global mdiff
    corr = get_norm_corr(alpha, beta)
    level = 1*(beta.get_volume()-0.5)

    mdiff = Shape()
    mdiff = mdiff.set_voxel(corr.get_sublevel_set(level))

    return mdiff

def minkowski_sum_and_diff(alpha, beta):
    global msum, mdiff

    corr = get_norm_corr(alpha, beta)
    level_sum = 0.0001*(beta.get_volume()-0.5)
    level_diff = 1*(beta.get_volume()-0.5)

    msum = Shape()
    mdiff = Shape()

    msum.set_voxel(corr.get_sublevel_set(level_sum))
    mdiff.set_voxel(corr.get_sublevel_set(level_diff))

    return msum, mdiff

def compute():
    global alpha, beta, msum, mdiff
    
    if (not alpha.isempty()) and (not beta.isempty()) :
        # Getting the total size for convolution
        sza = alpha.get_voxel_shape()
        szb = beta.get_voxel_shape()
        szc = sza + szb
        dims = [int(pow(2, ceil(log(dim)/log(2)))) for dim in szc]

        # Padding the shapes with zeros
        alpha.pad_voxel(dims)
        beta.pad_voxel(dims)

        msum, mdiff = minkowski_sum_and_diff(beta, alpha)

################################################################################

if __name__ == "__main__":
    
    # Initializing
    alpha = Shape()
    beta = Shape()
    msum = Shape()
    mdiff = Shape()
    
    app = QtGui.QApplication.instance()
    window = Window()
    window.show()

    app.exec_()
