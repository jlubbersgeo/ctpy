
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:49:35 2021

@author: Jordan Lubbers, Oregon State University

Last updated: 11/15/2021

CTPy is a collection of functions dedicated to the processing of Computed 
Tomography (CT) data in python. Built on top of scikit-image, it imports, 
denoises, plots, segments, and exports segmentation results. All figures are 
returned as matplotlib figure and axis objects and as such are fully 
customizable per matplotlib documentation. Data are processed as numpy arrays
and further mathematical/image processing operations can be utilized pending 
the research question under investigation. 

Happy processing!

"""
# bulk file import
import glob as glob

# for...warnings
import warnings

# visualization
import matplotlib.pyplot as plt

# array operations
import numpy as np

# saving metadata
import pandas as pd

# for loading dicom images
import pydicom as dicom
from PIL import Image

# image processing
from skimage import exposure, restoration
from skimage.filters import prewitt, roberts, scharr, sobel
from skimage.segmentation import watershed

# progress bar
from tqdm.notebook import tqdm


#%%
# helper function to import an entire folder of 2D images
def import_stack(filepath, filetype, name, downsample=False):
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
        type of file the images are. Choices are 'tif', 'png', 'jpg', or 'dicom'
    name : string
        name of the dataset you are working with. This will be utilized for 
        figure labeling and filesaving functions later on.
    downsample: int
        factor to downsample by. 

    Returns
    -------
    stack : ndarray
        3D numpy ndarray where the first two dimensions pertain to an individual
        image (e.g., slice) in the dataset and the last dimension pertains to 
        the slice number in the stack. Since numpy arrays are in [rows,columns,depth]
        format, this is analagous to [y,x,z]

    """
    if "png" in filetype:
        # creates a list of filepaths for each 2D images
        infiles = sorted(glob.glob("{}/*.png".format(filepath)))

    elif "tif" in filetype:
        # creates a list of filepaths for each 2D images
        infiles = sorted(glob.glob("{}/*.tif".format(filepath)))

    elif "jpg" in filetype:
        # creates a list of filepaths for each 2D images
        infiles = sorted(glob.glob("{}/*.jpg".format(filepath)))
        
    elif "dicom" in filetype:
        #this grabs every single item in the dirpath 
        all_files = glob.glob("{}/*".format(filepath))

        #this should remove all file extensions e.g. .png .tiff .xlsx
        #anything with a ".", but NOT FOLDERS
        infiles = sorted([d for d in all_files if '.' not in d])
        
    
    if "dicom" in filetype:
        #open the first dicom file and get the data as a numpy array
        im = dicom.dcmread(infiles[0]).pixel_array
        # create an 'empty' array filled with zeros, (Slices X width X length) is the shape of the array.
        # this is equivalent to (z, y, x) in cartesian space
        stack = np.zeros((len(infiles), im.shape[0], im.shape[1]))

    
    else:
   
        # open the first image to acquire image size
        im = Image.open(infiles[0])

        # create an 'empty' array filled with zeros, (Slices X width X length) is the shape of the array.
        # this is equivalent to (z, y, x) in cartesian space
        stack = np.zeros((len(infiles), im.size[1], im.size[0]))

    # loop the files, each iteration adds one slice layer into the array.
    i = 0
    print("Importing images")
    for imfile in tqdm(infiles, total = len(infiles), unit = 'files'):
        if "dicom" in filetype:
            im = dicom.dcmread(imfile).pixel_array
        else:
            
            im = Image.open(imfile)
            
        stack[i, :, :] = np.array(im)
        i += 1

    # force each image to have a shape comprised of two even numbers
    # such that we can at least downsample by a factor of 2

    # original rows and cols dimensions
    input_size_rows = stack.shape[1]
    input_size_cols = stack.shape[2]

    # make row and column dimensions even numbers
    if input_size_rows % 2 == 0:

        input_size_rows = input_size_rows

    else:
        input_size_rows = input_size_rows - 1

    if input_size_cols % 2 == 0:

        input_size_cols = input_size_cols

    else:

        input_size_cols = input_size_cols - 1

    # update the stack dimensions by trimming off a pixel
    # from either the rows or the columns (or both)
    stack = stack[:, :input_size_rows, :input_size_cols]

    # DOWNSAMPLING
    ## uses following example from scipy cookbook
    # https://scipython.com/blog/binning-a-2d-array-in-numpy/

    # note this will only downsample by common factors between
    # the rows and columns dimensions

    # here's a helper function to figure that out:
    def cf(num1, num2):
        common_factors = []
        for i in range(1, min(num1, num2) + 1):
            if num1 % i == num2 % i == 0:
                common_factors.append(i)
        return common_factors

    # downsampled rows and cols dimensions
    output_size_rows = stack.shape[1] // downsample
    output_size_cols = stack.shape[2] // downsample

    # array of potential downsampling factors based on input
    # shape
    common_factors = np.array(cf(input_size_rows, input_size_cols))

    # downsample if the specified factor is a potential common factor
    if downsample in common_factors:
        print("downsampling images")
        # bin size for rows and cols
        bin_size_rows = input_size_rows // output_size_rows
        bin_size_cols = input_size_cols // output_size_cols

        # create empty array to fill with downsampled data
        # this will have downsampled rows and cols but NOT
        # height

        stack_downsample = np.zeros(
            (stack.shape[0], output_size_rows, output_size_cols)
        )

        # loop through and complete downsampling for every slice and
        # add it to the empty array defined above
        for i in tqdm(range(stack.shape[0]), total = stack.shape[0], unit = 'images'):

            im = stack[i, :, :]
            im_downsample = (
                im.reshape(
                    (output_size_rows, bin_size_rows, output_size_cols, bin_size_cols)
                )
                .mean(-1)
                .mean(1)
            )

            stack_downsample[i, :, :] = im_downsample

        stack_downsample = stack_downsample[::downsample, :, :]

        print("your original stack dimensions are {}".format(stack.shape))
        print("your downsampled stack dimensions are {}".format(stack_downsample.shape))

        return stack_downsample, stack.shape

    elif downsample not in common_factors:
        warnings.warn(
            """You have chosen a downsampling factor that is not a common factor \n of your row 
            and column shapes and have therefore returned the original data. Please choose an 
            appropriate downsampling factor"""
        )
        print("your stack dimensions are {}".format(stack.shape))
        print("potential downsampling factors: {}".format(common_factors))
        return stack, stack.shape

    elif downsample is False and stack.ravel().shape > 125000000:
        warnings.warn(
            """please condsider downsampling your stack as it is very large and 
            image processing may become much more difficult pending your computer 
            capabilities. Check your Activity monitor for advice.
            """
        )
        return stack, stack.shape

    else:
        print("your stack dimensions are {}".format(stack.shape))

        return stack, stack.shape




#%%

# helper function to rescale the data between 0 and 1 and also trim
# it at upper and lower limits
def rescale_stack(lower_lim, upper_lim, stack):
    """
    rescale_stack uses lower and upper defined boundaries to apply contrast 
    contrast stretching to a desired stack. In brief, it uses percentiles as 
    the boundary conditions so that a small amount of pixel values do not 
    perturb the colormap, making small changes in value harder to see. Plots
    the results of the contrast stretching by comparing the grayscale values
    for a single slice using both a histogram and 2D image. 
    
    rescale_stack(lower_lim,upper_lim,stack,slice_number):
        
    Parameters
    ----------
    lower_lim : scalar
        lower limit of the rescale range in percent (e.g., .01)
    upper_lim : scalar
        upper limit of the rescale range in percent (e.g., 99.99)
    stack : ndarray
        dataset to apply the contrast stretching to. Must be a 3D array
    
    Returns
    -------
    stack_rescale : ndarray
        contrast stretched and rescaled dataset in the same shape as the input
        stack
    stack_rescaleArray1D : 1D array
        contrast stretched data flattened to 1D so it can be visualized in a 
        histogram

    """

    # find the pixel values of the upper and lower percentile limits
    plow, phigh = np.percentile(stack, (lower_lim, upper_lim))

    # use those values to rescale the data
    stack_rescale = exposure.rescale_intensity(stack, in_range=(plow, phigh))

    stack_normal = (stack_rescale - stack_rescale.min()) / (
        stack_rescale.max() - stack_rescale.min()
    )

    stack_normal1D = stack_normal.ravel()


    return plow, phigh, stack_normal, stack_normal1D

#%%
# helper function for denoising your stack with the non local means function
def denoise_stack(stack_rescale, patch_size, patch_distance):
    """
    denoise_stack applies the non-local-means filter from scikit-image restoration
    module (https://scikit-image.org/docs/dev/api/skimage.restoration.html) to 
    to the entire stack on a slice by slice basis.
    
    denoise_stack(stack_rescale,patch_size,patch_distance):

    Parameters
    ----------
    stack_rescale : ndarray
        data to be denoised. This can be a single image (e.g., 2D array) or 
        whole stack of images (e.g., 3D array)
    patch_size : int
        patch_size argument from scikit-image.restoration.denoise_nl_means:
        "Size of patches used for denoising."
    patch_distance : int
        patch_distance argument from scikit-image.restoration.denoise_nl_means:
        "Maximal distance in pixels where to search patches used for denoising."
    Returns
    -------
    nlm_array : ndarray
        3D array in the same shape as "stack_rescale" that contains the denoised
        pixel values
    nlm_array1D : 1D array
        1D array that contains the denoised pixel values. These values are the 
        same as nlm_array, just flattened so that they can be used for a histogram

    """
    stack_length = len(stack_rescale)
    nlm_smooth = []
    for i in tqdm(range(0, stack_length)):
        patch_kw = dict(
            patch_size=patch_size, patch_distance=patch_distance, multichannel=True
        )
        nlm_smooth.append(
            restoration.denoise_nl_means(stack_rescale[i], fast_mode=True, **patch_kw)
        )
    # convert from list to numpy array
    stack_nlm = np.asarray(nlm_smooth)
    stack_nlm_1D = stack_nlm.ravel()

    print("Your stack has been smoothed using the Skimage non-local means algorithm")
    print("patch size: {}".format(patch_size))
    print("patch distance: {}".format(patch_distance))

    return stack_nlm, stack_nlm_1D

#%%
def create_elevation_map(smoothed_stack, method):
    """
    create_elevation_map utilizes the sobel filter from skimage to create an
    elevation map based on the gradient at each pixel. It iterates through the 
    entire stack of data to create an individual elevation map for each image

    Parameters
    ----------
    smoothed_stack : ndarray
        3D denoised data array
    
    method : string
        'sobel', 'roberts', 'scharr', 'prewitt'. This determines the type of 
        edge operator to use in creating the elevation map. See example here:
        https://scikit-image.org/docs/dev/auto_examples/edges/plot_edge_filter.html

    Returns
    -------
    elevation_map : ndarray
        collection of elevation maps in same shape as input (3D ndarray)

    """
    elevation_map = np.zeros(smoothed_stack.shape)
    if method == "sobel":

        for i in tqdm(range(len(elevation_map))):
            elevation_map[i, :, :] = sobel(smoothed_stack[i, :, :])

    elif method == "roberts":
        for i in tqdm(range(len(elevation_map))):
            elevation_map[i, :, :] = roberts(smoothed_stack[i, :, :])

    elif method == "scharr":
        for i in tqdm(range(len(elevation_map))):
            elevation_map[i, :, :] = scharr(smoothed_stack[i, :, :])

    elif method == "prewitt":
        for i in tqdm(range(len(elevation_map))):
            elevation_map[i, :, :] = prewitt(smoothed_stack[i, :, :])

    return elevation_map


#%%
def add_markers(stack_nlm, phase_limits):
    """
    Parameters:
    ----------
    
    stack_nlm | ndarray
    denoised data array. This is used to create the right size ndarray
    for the marker array
    
    phase_limits | list
    list of values to be used as delimiters for the phase boundaries.
    For phases that are neither the most or least attenuating values
    will be a list:
    
    phase_limits = [[0.5],
                    [0.6, 0.7],
                    [0.84]]
                    
    This will set markers for three phases where pixel locations with values
    less than 0.5 will be used as a marker for phase 1, locations with values 
    between 0.6 and 0.7 will be used as a marker for phase 2, locations with
    values greater than 0.84 will be used as a marker for phase 3. 
    
    Returns:
    --------
    
    markers | ndarray
    
    marker array the same shape as the input array. All unspecified marker
    locations are 0
    
    """
    markers = np.zeros(stack_nlm.shape)

    for j in range(len(phase_limits)):
        print("filling markers for phase: {}".format(j + 1))

        if len(phase_limits[j]) == 1:

            if j == len(phase_limits) - 1:
                print("pixel boundary: x > {}".format(phase_limits[j]))
                for i in tqdm(range(len(markers))):
                    markers[i, :, :][stack_nlm[i] > phase_limits[j]] = j + 1
            else:
                print("pixel boundary: x < {}".format(phase_limits[j]))
                for i in tqdm(range(len(markers))):
                    markers[i, :, :][stack_nlm[i] < phase_limits[j]] = j + 1

        elif len(phase_limits[j]) == 2:
            print(
                "pixel boundary: {} > x > {}".format(
                    phase_limits[j][0], phase_limits[j][1]
                )
            )
            for i in tqdm(range(len(markers))):
                markers[i, :, :][
                    np.logical_and(
                        stack_nlm[i] > phase_limits[j][0],
                        stack_nlm[i] < phase_limits[j][1],
                    )
                ] = (j + 1)
    return markers

#%%
def run_watershed_segmentation(elevation_array, markers):
    """
    run_watershed_segmentation utilizes the watershed function from the segmentation
    module (https://scikit-image.org/docs/dev/api/skimage.segmentation.html) and 
    times the results
    
    run_watershed_segmentation(smoothed_stack,markers):

    Parameters
    ----------
    elevation_array : ndarray
        data to apply the watershed algorithm to. This is our elevation map
        3D matrix
    markers : ndarray
        marker array in the same shape as the "elevation_array" argument

    Returns
    -------
    ws_results : ndarray
        array in the same shape as the "elevation_array" array
        with values from "markers" as decided by the watershed
        algorithm

    """

    # containers
    segmentation = np.zeros(elevation_array.shape)
    for i in tqdm(range(len(elevation_array))):

        # fill segmentation array with watershed results for each slice
        segmentation[i, :, :] = watershed(elevation_array[i, :, :], markers[i, :, :])

    return segmentation

#%%
def save_seg_results(outpath, name, ws_results, cmap,phase_limits):
    """
    save_seg_results takes each segmented 2D image (array) and saves them as 
    individual images to a desired folder. 
    
    save_seg_results(outpath,name,ws_results):
    
    will save the images in the following format:
        "outpath/name_seg_results_z.tif" where "name" and "outpath" are strings
        input to the function and "z" is a 3 digit slice number 
        (i.e., 000, 001, 002)

    Parameters
    ----------
    outpath : string
        filepath to the folder where the images are to be saved
    name : string
        name for the dataset you are working with. Try to keep this consistent
        throughout the entire project.
    ws_results : ndarray
        results from the watershed segmentation
    phase_limits : list
        list of values to be used as delimiters for the phase boundaries.
        For phases that are neither the most or least attenuating values
        will be a list:
        
        phase_limits = [[0.5],
                        [0.6, 0.7],
                        [0.84]]
        This is the same input as the `add_markers` function
                        


    Returns
    -------
    None.

    """

    for i in tqdm(range(len(ws_results))):

        fig, axes = plt.subplots(nrows=1, ncols=1)
        ax0 = axes
        ax0.imshow(ws_results[i], cmap=cmap,vmin = 0, vmax = len(phase_limits)-1)  # choose your segmentation algorithm here
        ax0.axis("off")  # no axes so its just the image

        fig.set_size_inches(5, 5)
        fig.subplots_adjust(
            left=None, bottom=None, right=None, top=None, wspace=None, hspace=None
        )
        plt.savefig(
            "{}/{}_seg_results{}.tif".format(outpath, name, i), bbox_inches="tight"
        )
        plt.close()
    print("Your images have been saved to the following directory:\n{}".format(outpath))
    
#%%
def save_metadata(segmentation_metadata, phase_limits, outpath, filename):

    """
    For the reproducible segmentation of your CT volume, this will
    produce a .csv file with all the decisions made in the segmentation
    process.
    
    
    
    Parameters:
    -----------
    
    segmentation_metadata | dict
    dictionary organized as follows:
    
    segmentation_metadata = {
        "name": name,
        "original_stack_shape": [original_stack_shape],
        "processed_stack_shape": [stack.shape],
        "rescaled_percentile_values": [[lower_lim, upper_lim]],
        "denoise_patch_size": patch_size,
        "denoise_patch_distance": patch_distance,
        "elevation_map_algorithm": elevation_algorithm,
    }
    
    phase_limits | list
    list of values to be used as delimiters for the phase boundaries.
    For phases that are neither the most or least attenuating values
    will be a list:
    
    phase_limits = [[0.5],
                    [0.6, 0.7],
                    [0.84]]
                    
    This will set markers for three phases where pixel locations with values
    less than 0.5 will be used as a marker for phase 1, locations with values 
    between 0.6 and 0.7 will be used as a marker for phase 2, locations with
    values greater than 0.84 will be used as a marker for phase 3. 
    

    
    outpath | string
    the directory to save the metadata to
    
    filename| the name of the file being saved
    
    
    
    Returns:
    --------
    
    None
    
    *will save a csv with metadata in the form of a table to the 
    specified directory. Ex:
    
        Parameter|value
        ----------------------
        name|0009_20CJ04_x7_A
        ------------------------------------
        original_stack_shape|(544, 860, 872)
        ------------------------------------
        processed_stack_shape|(272, 430, 436)
        ------------------------------------
        rescaled_percentile_values|[0.05, 99.97]
        ------------------------------------
        denoise_patch_size|10
        ------------------------------------
        denoise_patch_distance|10
        ------------------------------------
        elevation_map_algorithm|sobel
        ------------------------------------
        phase 1 normalized pixel boundary|[0.5]
        ------------------------------------
        phase 2 normalized pixel boundary|[0.6, 0.67]
        ------------------------------------
        phase 3 normalized pixel boundary|[0.7, 0.83]
        ------------------------------------
        phase 4 normalized pixel boundary|[0.84]

    """

    
    for i, limit in enumerate(phase_limits):
        segmentation_metadata.update(
            {"phase {} normalized pixel boundary".format(i + 1): [limit]}
        )

    outfile = "{}_segmentation_metadata.csv".format(filename)

    df = pd.DataFrame(
        dict([(k, pd.Series(v)) for k, v in segmentation_metadata.items()])
    ).T.reset_index()
    df.columns = ["Parameter", "value"]
    df.to_csv(outpath + outfile, index=False)
    print(
        "Your metadata has been saved as:\n{}\n\nin the following directory: \n{}".format(
            outfile, outpath
        )
    )


