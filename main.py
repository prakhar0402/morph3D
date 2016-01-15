# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
# By default, the PySide binding will be used. If you want the PyQt bindings
# to be used, you need to set the QT_API environment variable to 'pyqt'
#os.environ['QT_API'] = 'pyqt'

# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
#from pyface.qt import QtGui, QtCore
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
import sip
sip.setapi('QString', 2)

from traits.api import HasTraits, Instance, on_trait_change
from traitsui.api import View, Item
from mayavi import mlab
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, \
        SceneEditor

from PyQt4 import QtGui, QtCore
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
        self.resolution = 64
        self.scale = 1
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
            if self.scale != 1:
                self.pad_voxel([self.resolution] * 3)

    def write_voxel(self, filename):
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
        self.voxel = voxel
        self.set_resolution(self.get_voxel_shape()[0])

    def set_voxel_ft(self, voxel_ft):
        self.voxel_ft = voxel_ft

    def set_size(self):
        self.size = max(1, int(self.scale * self.resolution))

    def set_resolution(self, resolution):
        self.resolution = resolution
        self.set_size()

    def set_scale(self, scale):
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
        beta.display(self.scene.mlab, contours = [1], color = (0, 0, 1), opacity = 0.5)
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

        self.ui = self.visualization.edit_traits(parent=self, kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)

################################################################################

################################################################################
# The Window class
class Window(QtGui.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.setWindowTitle("Morph3D")
        self.setStyleSheet('background-color: white')
        self.home()
        
    def home(self):
        self.container = QtGui.QWidget(self)
        
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create("Plastique"))
        self.layout = QtGui.QGridLayout(self.container)
        
        self.file_grid = QtGui.QGridLayout()
        self.res_grid = QtGui.QGridLayout()
        self.right_grid = QtGui.QGridLayout()
        
        self.morph = QtGui.QPushButton("COMPUTE", self.container)
        self.morph.clicked.connect(self.compute)
        self.morph.resize(self.morph.sizeHint())
        self.file_grid.addWidget(self.morph, 0, 0)
        
        self.part0 = QtGui.QPushButton("Click to select the first part (A).", self.container)
        self.part0.clicked.connect(self.file_open0)
        self.part0.resize(self.part0.sizeHint())
        self.file_grid.addWidget(self.part0, 1, 0)
        
        self.part1 = QtGui.QPushButton("Click to select the second part (B).", self.container)
        self.part1.clicked.connect(self.file_open1)
        self.part1.resize(self.part1.sizeHint())
        self.file_grid.addWidget(self.part1, 2, 0)
        
        self.layout.addLayout(self.file_grid, 0, 0)
        
        self.mayavi_widget = MayaviQWidget(self.container)
        self.layout.addWidget(self.mayavi_widget, 1, 0)
        
        self.res_grid = QtGui.QGridLayout()
        
        self.res_label = QtGui.QLabel(self.container)
        self.res_label.setText("Resolution")
        self.res_label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.res_label.resize(self.res_label.minimumSizeHint())
        self.res_grid.addWidget(self.res_label, 0, 0)
        
        self.res0 = QtGui.QSpinBox(self.container)
        self.res0.setPrefix("A: ")
        self.res0.setRange(16, 128)
        self.res0.setValue(64)
        self.res0.setSingleStep(16)
        self.res0.resize(self.res0.minimumSizeHint())
        self.res0.valueChanged[int].connect(self.res_change0)
        self.res_grid.addWidget(self.res0, 1, 0)
        
        self.res1 = QtGui.QSpinBox(self.container)
        self.res1.setPrefix("B: ")
        self.res1.setRange(16, 128)
        self.res1.setValue(64)
        self.res1.setSingleStep(16)
        self.res1.resize(self.res1.minimumSizeHint())
        self.res1.valueChanged[int].connect(self.res_change1)
        self.res_grid.addWidget(self.res1, 2, 0)
        
        self.layout.addLayout(self.res_grid, 0, 1)
        
        self.scale_label = QtGui.QLabel(self.container)
        self.scale_label.setText("Scale")
        self.scale_label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.scale_label.resize(self.scale_label.minimumSizeHint())
        self.right_grid.addWidget(self.scale_label, 0, 0)
        
        self.scale_grid = QtGui.QVBoxLayout()
        
        self.scale0 = QtGui.QDoubleSpinBox(self.container)
        self.scale0.setPrefix("A: ")
        self.scale0.setDecimals(2)
        self.scale0.setRange(0.05, 1)
        self.scale0.setValue(1)
        self.scale0.setSingleStep(0.1)
        self.scale0.resize(self.scale0.minimumSizeHint())
        self.scale0.valueChanged[float].connect(self.scale_change0)
        self.scale_grid.addWidget(self.scale0)
        
        self.scale1 = QtGui.QDoubleSpinBox(self.container)
        self.scale1.setPrefix("B: ")
        self.scale1.setDecimals(2)
        self.scale1.setRange(0.05, 1)
        self.scale1.setValue(1)
        self.scale1.setSingleStep(0.1)
        self.scale1.resize(self.scale1.minimumSizeHint())
        self.scale1.valueChanged[float].connect(self.scale_change1)
        self.scale_grid.addWidget(self.scale1)
        
        self.right_grid.addLayout(self.scale_grid, 1, 0)
        
        self.vis_label = QtGui.QLabel(self.container)
        self.vis_label.setText("Visibility")
        self.vis_label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.vis_label.resize(self.vis_label.minimumSizeHint())
        self.right_grid.addWidget(self.vis_label, 2, 0)
        
        self.vis_grid = QtGui.QVBoxLayout()
        
        self.part0_cb = QtGui.QCheckBox(r"Part A", self.container)
        self.part1_cb = QtGui.QCheckBox("Part B", self.container)
        self.msum_cb = QtGui.QCheckBox("A o+ B", self.container) #U\+2296
        self.mdiff_cb = QtGui.QCheckBox("A o- B", self.container) #U+2297
        
        self.part0_cb.setStyleSheet("background-color: gray; color: red")
        self.part1_cb.setStyleSheet("background-color: gray; color: blue")
        self.msum_cb.setStyleSheet("background-color: gray; color: yellow")
        self.mdiff_cb.setStyleSheet("background-color: gray; color: green")
        
        self.part0_cb.toggle()
        self.part1_cb.toggle()
        self.msum_cb.toggle()
        self.mdiff_cb.toggle()
        
        self.part0_cb.stateChanged.connect(self.part0_vis)
        self.part1_cb.stateChanged.connect(self.part1_vis)
        self.msum_cb.stateChanged.connect(self.msum_vis)
        self.mdiff_cb.stateChanged.connect(self.mdiff_vis)
        
        self.vis_grid.addWidget(self.part0_cb)
        self.vis_grid.addWidget(self.part1_cb)
        self.vis_grid.addWidget(self.msum_cb)
        self.vis_grid.addWidget(self.mdiff_cb)
        
        self.right_grid.addLayout(self.vis_grid, 3, 0)
        
        self.save_label = QtGui.QLabel(self.container)
        self.save_label.setText("Save")
        self.save_label.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.save_label.resize(self.save_label.minimumSizeHint())
        self.right_grid.addWidget(self.save_label, 4, 0)
        
        self.save_grid = QtGui.QVBoxLayout()
        
        self.save_msum = QtGui.QPushButton("A o+ B", self.container)
        self.save_msum.clicked.connect(self.save_sum)
        self.save_msum.resize(self.save_msum.sizeHint())
        self.save_grid.addWidget(self.save_msum)
        
        self.save_mdiff = QtGui.QPushButton("A o- B", self.container)
        self.save_mdiff.clicked.connect(self.save_diff)
        self.save_mdiff.resize(self.save_mdiff.sizeHint())
        self.save_grid.addWidget(self.save_mdiff)
        
        self.right_grid.addLayout(self.save_grid, 5, 0)
        
        self.layout.addLayout(self.right_grid, 1, 1)
        
        self.container.show()
        self.setCentralWidget(self.container)
        
    def file_open0(self):
        global alpha
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        self.part0.setText("Part A: " + name)
        alpha.set_filename(name)
        alpha.read_voxel()
        self.resetAll()
        
    def file_open1(self):
        global beta
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        self.part1.setText("Part B: " + name)
        beta.set_filename(name)
        beta.read_voxel()
        self.resetAll()
        
    def res_change0(self, value):
        global alpha
        alpha.set_resolution(value)
        self.reset()
        
    def res_change1(self, value):
        global beta
        beta.set_resolution(value)
        self.reset()
        
    def scale_change0(self, value):
        global alpha
        alpha.set_scale(value)
        self.reset()
        
    def scale_change1(self, value):
        global beta
        beta.set_scale(value)
        self.reset()
        
    def part0_vis(self):
        global alpha
        alpha.toggle_visibility()
        self.update()
        
    def part1_vis(self):
        global beta
        beta.toggle_visibility()
        self.update()
        
    def msum_vis(self):
        global msum
        msum.toggle_visibility()
        self.update()
        
    def mdiff_vis(self):
        global mdiff
        mdiff.toggle_visibility()
        self.update()
        
    def save_sum(self):
        global msum
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        msum.write_voxel(name)
        
    def save_diff(self):
        global mdiff
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        mdiff.write_voxel(name)
        
    def compute(self):
        compute()
        self.morph.setText('Done!')
        self.morph.setEnabled(False)
        self.update()
    
    def update(self):
        self.mayavi_widget.visualization.update_plot()
        
    def reset(self):
        self.morph.setText("COMPUTE")
        self.morph.setEnabled(True)
        
    def resetAll(self):
        global msum, mdiff
        self.morph.setText("COMPUTE")
        self.morph.setEnabled(True)
        msum = Shape()
        mdiff = Shape()
        msum.set_visibility(self.msum_cb.isChecked())
        mdiff.set_visibility(self.mdiff_cb.isChecked())
        self.update()

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
    msum.set_voxel(corr.get_sublevel_set(level))

def minkowski_diff(alpha, beta):
    global mdiff
    corr = get_norm_corr(alpha, beta)
    level = 1*(beta.get_volume()-0.5)
    mdiff.set_voxel(corr.get_sublevel_set(level))
    
def minkowski_sum_and_diff(alpha, beta):
    global msum, mdiff
    corr = get_norm_corr(alpha, beta)
    level_sum = 0.0001*(beta.get_volume()-0.5)
    level_diff = 1*(beta.get_volume()-0.5)
    msum.set_voxel(corr.get_sublevel_set(level_sum))
    mdiff.set_voxel(corr.get_sublevel_set(level_diff))

def compute():
    global alpha, beta, msum, mdiff
    alpha.read_voxel()
    beta.read_voxel()
    if (not alpha.isempty()) and (not beta.isempty()) :
        # Getting the total size for convolution
        sza = alpha.get_voxel_shape()
        szb = beta.get_voxel_shape()
        szc = sza + szb
        dims = [int(pow(2, ceil(log(dim)/log(2)))) for dim in szc]
        # Padding the shapes with zeros
        alpha.pad_voxel(dims)
        beta.pad_voxel(dims)
        # Computing the minkowski sum and difference
        minkowski_sum_and_diff(alpha, beta)

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
