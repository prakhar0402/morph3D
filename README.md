# Morph3D

**Morphology of 3D shapes**:
- Computes Minkowski sum (dilation)
- Computes Minkowski difference (erosion)
- Estimates manufacturability/non-manufacturability

This project provides a nice interface for *minkowski sum (dilation)* and *difference (erosion)* operations between two shapes. The interface takes .OBJ 3D surface mesh files as input. It converts the surface meshes into binary voxels before computing the sum and difference. The interface also provides functionality to set the resolution and scale of the input files. The visibility of the 3D shapes in the workspace can also be changed. The results can be saved as .BINVOX voxel files. Use `main.py` to run the interface. A couple of example input .OBJ files can be found in the directory `data/`.

In addition, the manufacturability of a shape (A) using a tool (B) can be predicted using the `main_as_man.py` file. It creates a similar interface, but results in regions of shape A that are manufacturable/non-manufacturable using the tool B. The methodology used for this computation is inspired by:
> Nelaturi, S., Burton, G., Fritz, C., & Kurtoglu, T. (2015, August). Automatic spatial planning for machining operations. In *Automation Science and Engineering (CASE), 2015 IEEE International Conference* (pp. 677-682). IEEE.

Please look into `requirements.txt` for the list of required libraries. The file `additional_step.txt` includes some instructions for installing few complex libraries. The interface was successfully tested on Linux. However, using it on Windows/Mac should also be straight forward.

*binvox credit: Patrick Min, http://www.cs.princeton.edu/~min/binvox/*

Please leave your comments and feedback at <prakharj@buffalo.edu>.

If you use our code or part of our code, please cite us.

MAD Lab, University at Buffalo
Copyright (C) 2018  Prakhar Jaiswal <prakharj@buffalo.edu>
