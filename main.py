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
            
################################################################################

################################################################################
# The Visualization class
class Visualization(HasTraits):
    scene = Instance(MlabSceneModel, ())

    @on_trait_change('scene.activated')
    def update_plot(self):
        '''
        Clears the current mayavi screen and redraws all four voxel models
        '''
        self.scene.mlab.clf()
        alpha.display(self.scene.mlab, contours = [1],
                        color = (1, 0, 0), opacity = 0.5)
        beta.display(self.scene.mlab, contours = [1],
                        color = (0, 0, 1), opacity = 0.5)
        msum.display(self.scene.mlab, contours = [1],
                        color = (1, 1, 0), opacity = 0.3)
        mdiff.display(self.scene.mlab, contours = [1],
                        color = (0, 1, 0), opacity = 0.3)

    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                height=600, width=800, show_label=False), resizable=True)
                     
# The MayaviQWidget class
class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        '''
        Creates mayavi sub-window as a widget in pyqt window
        '''
        QtGui.QWidget.__init__(self, parent)
        layout = QtGui.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        self.visualization = Visualization()

        self.ui = self.visualization.edit_traits(parent=self,
                                                    kind='subpanel').control
        layout.addWidget(self.ui)
        self.ui.setParent(self)
        
# The Window class
class Window(QtGui.QMainWindow):

    def __init__(self):
        '''
        Creates a PyQt window and call 'home' which creates all the objects
        inside the window
        '''
        super(Window, self).__init__()
        self.setWindowTitle("Morph3D")
        self.setStyleSheet('background-color: white')
        self.home()
        
    def home(self):
        '''
        Creates all the objects inside the window
        '''
        # container is the main central widget
        self.container = QtGui.QWidget(self)
        
        # setting Plastique as the default style
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create("Plastique"))
        
        # defining a grid layout
        self.layout = QtGui.QGridLayout(self.container)
        
        # subgrids inside the main grid layout
        self.file_grid = QtGui.QGridLayout()
        self.res_grid = QtGui.QGridLayout()
        self.right_grid = QtGui.QGridLayout()
        
        # The file subgrid at (0, 0)
        
        # push-button to invoke the computation of minkowski sum and difference
        self.morph = QtGui.QPushButton("COMPUTE", self.container)
        self.morph.setEnabled(False)
        self.morph.clicked.connect(self.compute)
        self.morph.resize(self.morph.sizeHint())
        self.file_grid.addWidget(self.morph, 0, 0)
        
        # push-button to select the file to read shape 'alpha' from
        self.partA = QtGui.QPushButton("Click to select the first part (A).",
                                                    self.container)
        self.partA.clicked.connect(self.file_openA)
        self.partA.resize(self.partA.sizeHint())
        self.file_grid.addWidget(self.partA, 1, 0)
        
        # push-button to select the file to read shape 'beta' from
        self.partB = QtGui.QPushButton("Click to select the second part (B).",
                                                    self.container)
        self.partB.clicked.connect(self.file_openB)
        self.partB.resize(self.partB.sizeHint())
        self.file_grid.addWidget(self.partB, 2, 0)
        
        # adding file subgrid to layout
        self.layout.addLayout(self.file_grid, 0, 0)
        
        # The mayavi sub-window at (1, 0)
        self.mayavi_widget = MayaviQWidget(self.container)
        self.layout.addWidget(self.mayavi_widget, 1, 0)
        
        # The resolution subgrid at (0, 1)
        
        # label for resolution        
        self.res_label = QtGui.QLabel(self.container)
        self.res_label.setText("Resolution")
        self.res_label.setAlignment(QtCore.Qt.AlignHCenter|
                                                QtCore.Qt.AlignBottom)
        self.res_label.resize(self.res_label.minimumSizeHint())
        self.res_grid.addWidget(self.res_label, 0, 0)
        
        # spin-box to set resolution for shape A
        self.resA = QtGui.QSpinBox(self.container)
        self.resA.setPrefix("A: ")
        self.resA.setRange(16, 128)
        self.resA.setValue(64)
        self.resA.setSingleStep(16)
        self.resA.resize(self.resA.minimumSizeHint())
        self.resA.valueChanged[int].connect(self.res_changeA)
        self.res_grid.addWidget(self.resA, 1, 0)
        
        # spin-box to set resolution for shape B
        self.resB = QtGui.QSpinBox(self.container)
        self.resB.setPrefix("B: ")
        self.resB.setRange(16, 128)
        self.resB.setValue(64)
        self.resB.setSingleStep(16)
        self.resB.resize(self.resB.minimumSizeHint())
        self.resB.valueChanged[int].connect(self.res_changeB)
        self.res_grid.addWidget(self.resB, 2, 0)
        
        # adding resolution subgrid to layout
        self.layout.addLayout(self.res_grid, 0, 1)
        
        # The right subgrid at (1, 1)
        
        # subsubgrids inside the right subgrid
        self.scale_grid = QtGui.QVBoxLayout()
        self.vis_grid = QtGui.QVBoxLayout()
        self.save_grid = QtGui.QVBoxLayout()
        
        # label for scale
        self.scale_label = QtGui.QLabel(self.container)
        self.scale_label.setText("Scale")
        self.scale_label.setAlignment(QtCore.Qt.AlignHCenter|
                                                QtCore.Qt.AlignBottom)
        self.scale_label.resize(self.scale_label.minimumSizeHint())
        self.right_grid.addWidget(self.scale_label, 0, 0)
        
        # spin-box to set scale for shape A
        self.scaleA = QtGui.QDoubleSpinBox(self.container)
        self.scaleA.setPrefix("A: ")
        self.scaleA.setDecimals(2)
        self.scaleA.setRange(0.05, 1)
        self.scaleA.setValue(1)
        self.scaleA.setSingleStep(0.1)
        self.scaleA.resize(self.scaleA.minimumSizeHint())
        self.scaleA.valueChanged[float].connect(self.scale_changeA)
        self.scale_grid.addWidget(self.scaleA)
        
        # spin-box to set scale for shape B
        self.scaleB = QtGui.QDoubleSpinBox(self.container)
        self.scaleB.setPrefix("B: ")
        self.scaleB.setDecimals(2)
        self.scaleB.setRange(0.05, 1)
        self.scaleB.setValue(1)
        self.scaleB.setSingleStep(0.1)
        self.scaleB.resize(self.scaleB.minimumSizeHint())
        self.scaleB.valueChanged[float].connect(self.scale_changeB)
        self.scale_grid.addWidget(self.scaleB)
        
        # adding scale subsubgrid to right subgrid
        self.right_grid.addLayout(self.scale_grid, 1, 0)
        
        # label for visibility
        self.vis_label = QtGui.QLabel(self.container)
        self.vis_label.setText("Visibility")
        self.vis_label.setAlignment(QtCore.Qt.AlignHCenter|
                                                    QtCore.Qt.AlignBottom)
        self.vis_label.resize(self.vis_label.minimumSizeHint())
        self.right_grid.addWidget(self.vis_label, 2, 0)
        
        
        #TODO: use the right operator symbols to display on GUI window
        
        # checkboxes for setting visibility for each of the four shapes
        self.partA_cb = QtGui.QCheckBox("Part A", self.container)
        self.partB_cb = QtGui.QCheckBox("Part B", self.container)
        self.msum_cb = QtGui.QCheckBox("A o+ B", self.container) #U\+2296
        self.mdiff_cb = QtGui.QCheckBox("A o- B", self.container) #U+2297
        
        self.partA_cb.setStyleSheet("background-color: gray; color: red")
        self.partB_cb.setStyleSheet("background-color: gray; color: blue")
        self.msum_cb.setStyleSheet("background-color: gray; color: yellow")
        self.mdiff_cb.setStyleSheet("background-color: gray; color: green")
        
        self.partA_cb.toggle()
        self.partB_cb.toggle()
        self.msum_cb.toggle()
        self.mdiff_cb.toggle()
        
        self.partA_cb.stateChanged.connect(self.partA_vis)
        self.partB_cb.stateChanged.connect(self.partB_vis)
        self.msum_cb.stateChanged.connect(self.msum_vis)
        self.mdiff_cb.stateChanged.connect(self.mdiff_vis)
        
        self.vis_grid.addWidget(self.partA_cb)
        self.vis_grid.addWidget(self.partB_cb)
        self.vis_grid.addWidget(self.msum_cb)
        self.vis_grid.addWidget(self.mdiff_cb)
        
        # adding visibility subsubgrid inside right subgrid
        self.right_grid.addLayout(self.vis_grid, 3, 0)
        
        # label for save
        self.save_label = QtGui.QLabel(self.container)
        self.save_label.setText("Save")
        self.save_label.setAlignment(QtCore.Qt.AlignHCenter|
                                                    QtCore.Qt.AlignBottom)
        self.save_label.resize(self.save_label.minimumSizeHint())
        self.right_grid.addWidget(self.save_label, 4, 0)
        
        # pushbotton to save the computed minkowski sum as binvox file
        self.save_msum = QtGui.QPushButton("A o+ B", self.container)
        self.save_msum.clicked.connect(self.save_sum)
        self.save_msum.resize(self.save_msum.sizeHint())
        self.save_grid.addWidget(self.save_msum)
        
        # pushbutton to save the computed minkowski difference as binvox file
        self.save_mdiff = QtGui.QPushButton("A o- B", self.container)
        self.save_mdiff.clicked.connect(self.save_diff)
        self.save_mdiff.resize(self.save_mdiff.sizeHint())
        self.save_grid.addWidget(self.save_mdiff)
        
        # adding save subsubgrid inside right subgrid
        self.right_grid.addLayout(self.save_grid, 5, 0)
        
        # adding right subgrid inside the main grid layout
        self.layout.addLayout(self.right_grid, 1, 1)
        
        # show and set the container as central widget in pyqt window
        self.container.show()
        self.setCentralWidget(self.container)
        
    def file_openA(self):
        '''
        Pushbutton 'partA' callback function
        Opens and reads in a triangulated file into a 3D voxel model 'alpha'
        '''
        global alpha
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        if len(name) != 0:
            self.partA.setText("Part A: " + name)
            alpha.set_filename(name)
            alpha.read_voxel()
            self.resetAll()
        
    def file_openB(self):
        '''
        Pushbutton 'partB' callback function
        Opens and reads in a triangulated file into a 3D voxel model 'beta'
        '''
        global beta
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        if len(name) != 0:
            self.partB.setText("Part B: " + name)
            beta.set_filename(name)
            beta.read_voxel()
            self.resetAll()
        
    def res_changeA(self, value):
        '''
        Spin-box 'resA' callback function
        Changes the resolution for shape 'alpha'
        '''
        global alpha
        alpha.set_resolution(value)
        self.reset()
        
    def res_changeB(self, value):
        '''
        Spin-box 'resB' callback function
        Changes the resolution for shape 'beta'
        '''
        global beta
        beta.set_resolution(value)
        self.reset()
        
    def scale_changeA(self, value):
        '''
        Spin-box 'scaleA' callback function
        Changes the scale for shape 'alpha'
        '''
        global alpha
        alpha.set_scale(value)
        self.reset()
        
    def scale_changeB(self, value):
        '''
        Spin-box 'scaleB' callback function
        Changes the scale for shape 'beta'
        '''
        global beta
        beta.set_scale(value)
        self.reset()
        
    def partA_vis(self):
        '''
        Checkbox 'partA_cb' callback function
        Toggles the visibility of shape 'alpha'
        '''
        global alpha
        alpha.toggle_visibility()
        self.update()
        
    def partB_vis(self):
        '''
        Checkbox 'partB_cb' callback function
        Toggles the visibility of shape 'beta'
        '''
        global beta
        beta.toggle_visibility()
        self.update()
        
    def msum_vis(self):
        '''
        Checkbox 'msum_cb' callback function
        Toggles the visibility of shape 'msum'
        '''
        global msum
        msum.toggle_visibility()
        self.update()
        
    def mdiff_vis(self):
        '''
        Checkbox 'mdiff_cb' callback function
        Toggles the visibility of shape 'mdiff'
        '''
        global mdiff
        mdiff.toggle_visibility()
        self.update()
        
    def save_sum(self):
        '''
        Pushbutton 'save_msum' callback function
        Prompts the user to save shape 'msum' as binvox file
        '''
        global msum
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        msum.write_voxel(name)
        
    def save_diff(self):
        '''
        Pushbutton 'save_mdiff' callback function
        Prompts the user to save shape 'mdiff' as binvox file
        '''
        global mdiff
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        mdiff.write_voxel(name)
        
    def compute(self):
        '''
        Pushbutton 'morph' callback function
        Computes the minkowski sum and difference and displays it
        '''
        compute()
        self.morph.setText('Done!')
        self.morph.setEnabled(False)
        self.update()
    
    def update(self):
        '''
        Updates the mayavi subwindow display
        '''
        self.mayavi_widget.visualization.update_plot()
        
    def reset(self):
        '''
        Resets the pushbutton 'morph' to enabled state
        '''
        self.morph.setText("COMPUTE")
        self.morph.setEnabled(True)
        
    def resetAll(self):
        '''
        Resets the pushbutton 'morph' to enabled state and reinitializes the
        shapes msum and mdiff
        '''
        global msum, mdiff
        self.reset()
        msum = Shape()
        mdiff = Shape()
        msum.set_visibility(self.msum_cb.isChecked())
        mdiff.set_visibility(self.mdiff_cb.isChecked())
        self.update()

################################################################################
# Global functions

def get_norm_corr(alpha, beta):
    '''
    Computes normalized convolution of two shapes (3D rasterized models)
    Uses Fast Fourier Transform (FFT) to compute convolution efficiently
    Input: 'alpha' and 'beta' - Instances of class 'Shape()'
    Output: 'corr' - Instance of class 'Shape()'
    '''
    # taking Fourier tranform of shapes 'alpha' and 'beta'
    alpha.fourier_transform()
    beta.fourier_transform()
    
    # setting 'corr.voxel_ft' as product of fourier transforms of two shapes
    corr = Shape()
    corr.set_voxel_ft(alpha.get_voxel_ft() * beta.get_voxel_ft())
    
    # computing inverse Fourier transform for 'corr' and normalizing it
    corr.inverse_fourier_transform()
    corr.normalize()

    return corr

def minkowski_sum(alpha, beta):
    '''
    Computes minkowski sum using convolution algebra
    Input: 'alpha' and 'beta' - Instances of class 'Shape()'
    Output stored in global variables 'msum'
    '''
    global msum
    
    # getting the convolution of two shapes
    corr = get_norm_corr(alpha, beta)
    
    # minkowski sum would set of all cells with positive value
    # hence, using a small number of (0.01% of volume of shape 'beta')
    # to mitigate the precision error
    level = 0.0001*(beta.get_volume()-0.5)
    
    # computing minkowski sum as sublevel set of convolution
    msum.set_voxel(corr.get_sublevel_set(level))

def minkowski_diff(alpha, beta):
    '''
    Computes minkowski difference using convolution algebra
    Input: 'alpha' and 'beta' - Instances of class 'Shape()'
    Output stored in global variables 'mdiff'
    '''
    global mdiff
    
    # getting the convolution of two shapes
    corr = get_norm_corr(alpha, beta)
    
    # minkowski difference would set of all cells with value larger than
    # volume of 'beta'
    # using a number slightly smaller than the volume of shape 'beta'
    # to mitigate the precision error
    level = 1*(beta.get_volume()-0.5)
    
    # computing minkowski sum as sublevel set of convolution
    mdiff.set_voxel(corr.get_sublevel_set(level))
    
def minkowski_sum_and_diff(alpha, beta):
    '''
    Computes minkowski sum and difference using convolution algebra
    Input: 'alpha' and 'beta' - Instances of class 'Shape()'
    Output stored in global variables 'msum' and 'mdiff'
    '''
    global msum, mdiff
    
    # getting the convolution of two shapes
    corr = get_norm_corr(alpha, beta)
    
    # minkowski sum would set of all cells with positive value
    # hence, using a small number of (0.01% of volume of shape 'beta')
    # to mitigate the precision error
    level_sum = 0.0001*(beta.get_volume()-0.5)
    
    # minkowski difference would set of all cells with value larger than
    # volume of 'beta'
    # using a number slightly smaller than the volume of shape 'beta'
    # to mitigate the precision error
    level_diff = 1*(beta.get_volume()-0.5)
    
    # computing minkowski sum and difference as sublevel sets of convolution
    msum.set_voxel(corr.get_sublevel_set(level_sum))
    mdiff.set_voxel(corr.get_sublevel_set(level_diff))

def compute():
    '''
    Computes the minkowski sum and difference.
    Global variables msum and mdiff are used to store the result.
    The operations are performed on shapes stored in global variable alpha and beta.
    'alpha', 'beta', 'msum', and 'mdiff' are all objects of class 'Shape()'.
    '''
    global alpha, beta, msum, mdiff
    
    alpha.read_voxel()
    beta.read_voxel()
    if (not alpha.isempty()) and (not beta.isempty()):
    
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
    
    # Initializing shapes
    alpha = Shape()
    beta = Shape()
    msum = Shape()
    mdiff = Shape()
    
    app = QtGui.QApplication.instance()
    window = Window()
    window.show()

    app.exec_()
