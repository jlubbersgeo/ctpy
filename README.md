# CTPy
*CT data processing in Python*

[![DOI](https://zenodo.org/badge/338938387.svg)](https://zenodo.org/badge/latestdoi/338938387)


![png](Ct_processing_diagram.png)


## Overview
```CTPy``` is a collection of Python functions that helps go from a stack of 2D tomographic images that represent a 3D volume to that same volume segmented into user defined regions based on their grayscale value. Built on top of [scikit-image](https://scikit-image.org/), the functions help the user achieve the following:


- **Load in stack of image files that represent slices through the dataset:** This is your 'raw' data

- **Resize your dataset** This is especially useful if you are dealing with high resolution stacks of data on a personal computer

- **Normalize the stack values between 0 and 1:** scikit image algorithms work better this way

- **Apply contrast stretching:** This removes small, high attenuating areas that don't reflect the mineral (e.g. small oxide inclusions). Allows for us to see subtleties in mineral zone attenuation better.
    
- **Denoise the data using a non-local means filter:** Creates more distinct histogram peaks while preserving textural information

- **Choose the [markers](https://www.cmm.mines-paristech.fr/~beucher/wtshed.html) for the watershed segmentation algorithm:** These help prevent oversegmentation and more accurate image classification
- **Apply [watershed algorithm](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html) to entire stack:** Applies same markers for every image
- **Save segmented stack to folder:** This is done as a consecutive series of 2D images similar to imageJ

For more information on each function just use the standard Python documentation for learning about a function:
```python
help(ct.import_stack)

```
```

    """
    import_stack imports a stack of images and converts them into a 3D numpy 
    array so that it can be further processed using either numpy functions or
    scikit-image based functions in the CTPy functions in the module.
    
    import_stack(filepath,filetype,name):
        
    Parameters
    ----------
    filepath : string
        file directory for where the series of images is located. It is recommended
        that the files be in their own folder, as the function will import all 
        files from the directory as specified in the 'filetype' argument. 
    filetype : string
        type of file the images are. Choices are 'tif', 'png', or 'jpg'.
    name : string
        name of the dataset you are working with. This will be utilized for 
        figure labeling and filesaving functions later on.
    rescale: float
        factor to downsample by. .5 is downsampling by factor of 2, .1 is factor 
        of 10, etc.
    Returns
    -------
    stack : ndarray
        3D numpy ndarray where the first two dimensions pertain to an individual
        image (e.g., slice) in the dataset and the last dimension pertains to 
        the slice number in the stack. Since numpy arrays are in [rows,columns,depth]
        format, this is analagous to [y,x,z]
    """

```
## Installation and Usage
```CTPy``` can be installed locally and run by creating a virtual environment. If you are new to python, we recommend doing this through [Anaconda](https://www.anaconda.com/products/individual).

```
git clone https://github.com/jlubbersgeo/ctpy
cd /path/to/ctpy
conda create -n ctpy
conda activate ctpy
conda install --file requirements.txt
```

It is recommended that ```CTPy``` is utilized in conjunction with the Jupyter ecosystem, as many of the functions require the user to make decisions based results of prior figures/functions. For an example of how to utilize ```CTPy``` in your own work, please see the ```mineral_segmentation.ipynb``` example.


From now on any time you wish to use ctpy, simply re-activate the virtual environment. All the requisite packages are still installed. 


### Caveats

On windows you may need to add the following channel for downloading the ```requirements.txt```file:
```
git clone https://github.com/jlubbersgeo/ctpy
cd /path/to/ctpy
conda create -n ctpy
conda activate ctpy
conda config --append channels conda-forge
conda install --file requirements.txt
```

