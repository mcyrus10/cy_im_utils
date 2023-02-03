# Napari Notes

Some of the gui stuff that I have done here uses the napari nd image viewer
with the magicgui to make widgets etc. Here I plan to outline some of the bugs
and expected workflow to help the user (most likely me)

## unwrap.py

The unwrapping gui is a bit more temperamental than the tomography
reconstruction gui. It doesn't require a config file, so you can just launch it
with
    $ python unwrap.py
This will launch napari with the widgets all on the right. 
The process for unwrapping is as follows:
1. Load a reconstruction into napari
    - file -> open files as stack
2. View the volume and find the coordinates where you want to begin the
   unwrapping
3. Create a new shapes layer  $\rightarrow$ path
4. Trace the part of the reconstruction that traces the unwrapping
    - this is a sort of inexact science that I haven't yet masetered, but it
      seems like slightly denser points might work better
5. Click "fit spline" (on the right)
6. Click "show spline" (on the right)
    - layer number can be -1 to select the current layer or the layer number
7. Find the termination layer of the unwrapping
8. Enter that layer into the "destination layer" on the right then click 'clone
   point path'
9. Now you can select the point path on the second layer and modify its
   vertices to trace the same spiral in this layer
10. repeat 5 and 6 on this layer
11. If you are satisfied with the splines select the sampling then click
    "unwrap volume"
    - note the sampling is how many pixels on each side of the spline to
      sample, so the unwrapped volume will have twice as many pixels as
      sampling
12. Inspect volume in the viewer and iterate on refining the splines, etc.
