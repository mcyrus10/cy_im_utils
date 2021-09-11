# Cyrus Image Utils

This should mostly just be functions, unless a class is really appropriate, it
will be simpler most of the time to just deal with numpy arrays.

**Default dtype : np.float32**

The most natural way to represent images as 3D arrays?
    index 1 = Image index (in the stack)
    index 2 = Column
    index 3 = Row

Need to think about a consistent image representation!

Probably try to stick with a 3-tier hierarchy like SAREPY:

cy_im_utils is the root

branch tier 1:
    - Prep? -> All high performance
        - GPU data reduction pipeline? (Radiographs -> Attenuation?)
        - Field
        - Imread -> Dask?
        - extract norm patch
    - Extract?
        - patch_slice
    - Visualization
        - template for an interactive plot
        - patch plotting
        - nif_99to01contrast
    - statistical testing
    - Patch_extract


from cy_im_utils.analysis.visualization import nif_99to01contrast
from cy_im_utils.prep import field
