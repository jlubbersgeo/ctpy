#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 17:49:35 2021

@author: Jordan Lubbers, Oregon State University

Last updated: 2/14/2021

CTPy is a collection of functions dedicated to the processing of Computed 
Tomography (CT) data in python. Built on top of scikit-image, it imports, 
denoises, plots, segments, and exports segmentation results. All figures are 
returned as matplotlib figure and axis objects and as such are fully 
customizable per matplotlib documentation. Data are processed as numpy arrays
and further mathematical/image processing operations can be utilized pending 
the research question under investigation. 

Happy processing!

"""
import matplotlib.pyplot as plt 


#array operations
import numpy as np

#importing data
import glob as glob
from PIL import Image

#progress bar
from tqdm.notebook import tqdm

#image processing
from skimage import restoration
from skimage.segmentation import watershed 
from skimage import exposure
from skimage.filters import sobel

import time 

# helper function to import an entire folder of 2D images
def import_stack(filepath,filetype,name):
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

    Returns
    -------
    stack : ndarray
        3D numpy ndarray where the first two dimensions pertain to an individual
        image (e.g., slice) in the dataset and the last dimension pertains to 
        the slice number in the stack. Since numpy arrays are in [rows,columns,depth]
        format, this is analagous to [y,x,z]

    """
    if 'png' in filetype:  
        # creates a list of filepaths for each 2D images
        infiles = glob.glob('{}/*.png'.format(filepath))
    
    elif 'tif' in filetype:
        # creates a list of filepaths for each 2D images
        infiles = glob.glob('{}/*.tif'.format(filepath))
    
    elif 'jpg' in filetype:
        # creates a list of filepaths for each 2D images
        infiles = glob.glob('{}/*.jpg'.format(filepath))    


    #open the first image to acquire image size
    im = Image.open(infiles[0])

    #create an 'empty' array filled with zeros, (Slices X width X length) is the shape of the array.
    stack = np.zeros((len(infiles),im.size[1],im.size[0]))

    #loop the files, each iteration adds one slice layer into the array.
    i = 0
    for imfile in tqdm(infiles):
        im = Image.open(imfile)
        stack[i,:,:] = np.array(im)
        i+=1

    
    
    stack_length = len(infiles)
    print('your stack is ',stack_length, ' images thick and is ready to denoise')
    #Now, all values of the images are imported into the 3D array named 'stack', thus the array 'stack' 
    #is equivalent to the digitised real object
    return stack

# helper function to rescale the data between 0 and 1 and also trim
# it at upper and lower limits
def rescale_stack(lower_lim,upper_lim,stack,slice_number,cmap):
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
    slice_number : int
        slice number in the stack to be visualized for comparison between 
        original and contrast stretched data
    cmap : string
        any valid matplotlib colormap

    Returns
    -------
    stack_rescale : ndarray
        contrast stretched and rescaled dataset in the same shape as the input
        stack
    stack_rescaleArray1D : 1D array
        contrast stretched data flattened to 1D so it can be visualized in a 
        histogram
    fig : matplotib figure object
    ax : matplotlib axis object 

    """
    
    #find the pixel values of the upper and lower percentile limits
    plow,phigh = np.percentile(stack,(lower_lim,upper_lim))

    #use those values to rescale the data
    stack_rescale = exposure.rescale_intensity(stack,in_range=(plow,phigh))
    

    #Histogram of your whole stack rescaled between 0 and 1
    stackArray1D = stack.ravel()


    # plot up the comparison between original stack and rescaled stack
    fig, ax, = plt.subplots(2,2,figsize = (10,10))

    ax[0,0].hist(stackArray1D,
                 bins = 100,
                 color = 'gray'
                )
    ax[0,0].set_title('Stack Histogram',fontsize = 20)
    ax[0,0].set_yscale('log')
    ax[0,0].axvline(phigh,c = 'r',label = '{}% of data'.format(upper_lim))
    ax[0,0].legend(loc = 'upper left')
    ax[0,0].set_ylabel('counts',fontsize = 14)
    ax[0,0].set_xlabel('pixel value',fontsize = 14)
    
    #show the first image of the stack
    ax[0,1].imshow(stack[slice_number],cmap=cmap)  
    ax[0,1].set_title('Slice {} of stack'.format(slice_number))

    #rescaled data
    stack_rescaleArray1D = stack_rescale.reshape(-1)
    ax[1,0].hist(stack_rescaleArray1D,
                 bins = 100,
                 color = 'gray'
                )
    ax[1,0].set_yscale('log')
    ax[1,0].set_ylabel('counts',fontsize = 14)
    ax[1,0].set_xlabel('rescaled pixel value',fontsize = 14)
    ax[1,0].set_title('Normalized and rescaled stack',fontsize = 20)
    #show the first image of the stack
    ax[1,1].imshow(stack[slice_number],cmap=cmap)  
    ax[1,1].set_title('Slice {} of rescaled stack'.format(slice_number))
    fig.tight_layout()
    
    return stack_rescale, stack_rescaleArray1D, fig, ax 

# helper function for denoising your stack with the non local means function
def denoise_stack(stack_rescale,patch_size,patch_distance):
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
    for i in tqdm(range(0,stack_length)):
        patch_kw = dict(patch_size=patch_size,patch_distance=patch_distance,multichannel=True)
        nlm_smooth.append(restoration.denoise_nl_means(stack_rescale[i],  fast_mode=True,**patch_kw)) 
    # convert from list to numpy array
    nlm_array = np.asarray(nlm_smooth)
    nlm_array1D = nlm_array.ravel()

    print('Your stack has been smoothed using the Skimage non-local means algorithm')
    
    return nlm_array, nlm_array1D


# helper function to plot raw slice data and denoised slice data with their histograms
def denoise_slice_plot(cmap,bins,slice_number,rescaled_stack,smoothed_stack,name):
    """
    denoise_slice_plot plots the results of the "denoise_stack" function for a 
    desired slice in a similar arrangement to the "rescale_stack" function, where
    both a 2D image and histogram for the 2D image are plotted for the original
    and denoised slice.
    
    denoise_slice_plot(cmap,bins,slice_number,rescaled_stack,smoothed_stack,name):

    Parameters
    ----------
    cmap : string
        any valid matplotlib colormap (e.g., viridis, plasma, magma, etc.)
    bins : int
        number of bins to be displayed in the histograms
    slice_number : int
        slice number in the stack to be visualized for comparison between 
        original and denoised data
    rescaled_stack : ndarray
        stack of raw/rescaled data to be compared to the denoised data
    smoothed_stack : ndarray
        stack of denoised data from the "denoise_stack" function. "rescaled_stack"
        and "smoothed_stack" must be the same shape
    name : string
       name for the dataset you are working with. Try to keep this consistent
       throughout the process. 

    Returns
    -------
    fig : matplotib figure object
    ax : matplotlib axis object 

    """
    colormap = plt.get_cmap(cmap)
    nlm_smoothArray1D = smoothed_stack[slice_number].reshape(-1)
    slice_Array1D = rescaled_stack[slice_number].reshape(-1)


    fig, ax = plt.subplots(2,2, figsize = (15,10),gridspec_kw = {'width_ratios':[2,1],})
    fig.suptitle('Comparing slice {} before and after denoising \n {}'.format(slice_number,name), fontsize=18)

    #Histogram for rescaled image
    counts, bins, patches = ax[0,0].hist(slice_Array1D, bins = bins,)
    ax[0,0].minorticks_on()
    ax[0,0].set_xlim(0,1)
    ax[0,0].set_ylabel('counts',fontsize = 20)
    ax[0,0].set_title('Rescaled Image',fontsize = 20,y = .85)

    #get the value for center of bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    #color the histogram by value
    for c, p in zip(bin_centers,patches):
        plt.setp(p,'facecolor',colormap(c))

    #rescaled image
    ax[0,1].imshow(rescaled_stack[slice_number],cmap=cmap)
    ax[0,1].axes.yaxis.set_ticklabels([])
    ax[0,1].axes.yaxis.set_ticks([])
    ax[0,1].axes.xaxis.set_ticklabels([])
    ax[0,1].axes.xaxis.set_ticks([])   

    #Histogram for non local means image
    counts2, bins2, patches2, = ax[1,0].hist(nlm_smoothArray1D, bins = bins,)

    #get value for center of bins
    bin_centers2 = 0.5 * (bins[:-1] + bins[1:])

    #color the histogram by value
    for c, p in zip(bin_centers2,patches2):
        plt.setp(p,'facecolor',colormap(c))
    ax[1,0].minorticks_on()
    ax[1,0].set_ylim(0,1.05*np.max(counts2))
    ax[0,0].set_ylim(0,1.05*np.max(counts2))

    ax[1,0].set_xlim(0,1)
    ax[1,0].set_xlabel('normalized pixel value',fontsize = 14)
    ax[1,0].set_title('Non-local Means image', fontsize = 20, y = .85)   
    ax[1,0].set_ylabel('counts',fontsize = 20)


    #non local means image   
    ax[1,1].imshow(smoothed_stack[slice_number],cmap = cmap)
    ax[1,1].axes.yaxis.set_ticklabels([])
    ax[1,1].axes.yaxis.set_ticks([])
    ax[1,1].axes.xaxis.set_ticklabels([])
    ax[1,1].axes.xaxis.set_ticks([])
    fig.tight_layout()
    
    return fig, ax
    
    
    
    
# helper function for plotting up histograms of raw stack and denoised stack   
def denoise_stack_plot(cmap,bins,rescaled_stack1D,smoothed_stack1D):
    """
    denoise_stack_plot plots the results of the "denoise_stack" function for 
    the entire stack in the form of a histogram for the datset before and after
    denoising
    
    denoise_stack_plot(cmap,bins,rescaled_stack1D,smoothed_stack1D):

    Parameters
    ----------
    cmap : string
        any valid matplotlib colormap (e.g., viridis, plasma, magma, etc.)
    bins : int
        number of bins to be displayed in the histograms
    rescaled_stack1D : 1D array
        1D array pertaining to the original or rescaled dataset that was the 
        input for the "denoise_stack" function
    smoothed_stack1D : 1D array
        1D array pertaining to the results of the denoised stack.

    Returns
    -------
    fig : matplotib figure object
    ax : matplotlib axis object 

    """
    colormap = plt.get_cmap(cmap)
    
    fig, ax = plt.subplots(1,2,figsize = (12,4))

    counts, bins, patches = ax[0].hist(rescaled_stack1D,
                                       bins = bins,

    )
    ax[0].set_title('Noisy Stack Histogram',fontsize = 20)
    ax[0].set_xlabel('normalized pixel value',fontsize = 20)
    ax[0].set_ylabel('counts',fontsize = 20)
    ax[0].minorticks_on()


    counts2, bins2, patches2 = ax[1].hist(smoothed_stack1D,
                                          bins = bins,


    )

    #make histograms colored by normalized pixel value
    #first histogram
    #get the value for the center of the bins
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    #color the histogram by value
    for c, p in zip(bin_centers,patches):
        plt.setp(p,'facecolor',colormap(c))

    #second histogram
    bin_centers2 = 0.5 * (bins[:-1] + bins[1:])

    #color the histogram by value
    for c, p in zip(bin_centers2,patches2):
        plt.setp(p,'facecolor',colormap(c))

    ax[1].set_title('Denoised Stack Histogram',fontsize = 20)
    ax[1].set_xlabel('normalized pixel value')
    ax[1].minorticks_on()
    # ax[0].set_yscale('log')
    # ax[1].set_yscale('log')
    ax[0].set_ylim(0,np.max(1.05*counts2))
    ax[1].set_ylim(0,np.max(1.05*counts2))
    ax[0].set_xlim(0,1)
    ax[1].set_xlim(0,1)
    
    return fig, ax


def create_elevation_map(smoothed_stack):
    """
    create_elevation_map utilizes the sobel filter from skimage to create an
    elevation map based on the gradient at each pixel. It iterates through the 
    entire stack of data to create an individual elevation map for each image

    Parameters
    ----------
    smoothed_stack : ndarray
        3D denoised data array

    Returns
    -------
    elevation_map : ndarray
        collection of elevation maps in same shape as input (3D ndarray)

    """
    elevation_map = np.zeros(smoothed_stack.shape)
    
    for i in tqdm(range(len(elevation_map))):
        elevation_map[i,:,:] = sobel(smoothed_stack[i,:,:])
        
        
    return elevation_map



def plot_elevation_map(elevation_array,slice_number,cmap):
    """
    plot_elevation_map plots the sobel filter elevation map for a given slice in
    your data

    Parameters
    ----------
    elevation_array : ndarray
        3D elevation map array. This is the output from the "create_elevation_map"
        function
    slice_number : int
        slice in the stack you want to visualize
    cmap : string
        any valid matplotlib colormap to be used for visualizing the data 

    Returns
    -------
    fig : matplotlib figure object
       
    ax : matplotlib axis object
    
    """

    fig, ax = plt.subplots(figsize = (6,6))
    
    m = ax.imshow(elevation_array[slice_number],cmap = cmap)
    ax.set_title('Elevation Map \n slice number: {}'.format(slice_number),fontsize = 20)
    
    cbar = fig.colorbar(m,ax = ax,shrink = .75,)
    cbar.set_label(label = 'normalized gradient',fontsize = 16)
    
    ax.axis('off')
    
    return fig, ax

# helper function to plot up markers and denoised data for a given slice
def plot_markers(markers,smoothed_stack,slice_number,cmap):
    """
    plot_markers compares the markers established for the watershed algorithm
    with the denoised data by plotting the two images side by side 
    
    plot_markers(markers,smoothed_stack,slice_number,cmap):
        
    Parameters
    ----------
    markers : ndarray
        3D array in the same shape as the denoised data that contains marker values
        for the watershed algorithm
    smoothed_stack : ndarray
        3D array representing the denoised stack of data. "markers" and "smoothed_stack"
        must be the same shape
    slice_number : int
        slice number to be visulaized in the comparison
    cmap : string
        any valid matplotlib colormap

    Returns
    -------
    fig : matplotib figure object
    ax : matplotlib axis object 

    """
    fig, ax = plt.subplots(1,2,figsize = (12,6))
    m = ax[0].imshow(markers[slice_number],cmap = cmap)
    ax[0].set_title("Markers",fontsize = 20)
    bounds = np.linspace(0,5,6)

    cbar = fig.colorbar(m,ax = ax[0],ticks = bounds,shrink = .6,)
    cbar.set_label(label = 'marker value',fontsize = 16)
    ax[1].imshow(smoothed_stack[slice_number],cmap = cmap)
    ax[1].set_title("Denoised Data",fontsize =20)

    fig.tight_layout()
    
    return fig, ax

# helper function to run the watershed algorithm and time it
def run_watershed_segmentation(elevation_array,markers):
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
    
    #containers
    segmentation = np.zeros(elevation_array.shape)
    for i in tqdm(range(len(elevation_array))):
        
        #fill segmentation array with watershed results for each slice
        segmentation[i,:,:] = watershed(elevation_array[i,:,:], markers[i,:,:])
    
    return segmentation

# helper function for plotting up smoothed data, markers, and watershed results
def plot_ws_results(smoothed_stack,markers,ws_results,slice_number,cmap):
    """
    plot_ws_results compares the denoised data, markers for the watershed algorthim,
    and the results for the watershed algorthim in a 1x3 panel plot.
    
    plot_ws_results(smoothed_stack,markers,ws_results,slice_number,cmap):
        
    Parameters
    ----------
    smoothed_stack : ndarray
        stack that the watershed algorithm was applied to
    markers : ndarray
        marker array in the same shape as "smoothed_stack" used to train the 
        watershed algorithm
    ws_results : ndarray
        results from the watershed segmentation
        
    slice_number : int
        number pertaining to the individual slice you want to visualize from 
        the stack
    cmap : string
        any valid matplotlib colormap

    Returns
    -------
    fig : matplotib figure object
    ax : matplotlib axis object 

    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(smoothed_stack[slice_number],cmap = cmap, interpolation='nearest')
    ax[0].axis('off')
    ax[0].set_title('Smoothed data',fontsize = 20)
    ax[1].imshow(markers[slice_number],cmap = cmap, interpolation='nearest')
    ax[1].axis('off')
    ax[1].set_title('Markers',fontsize = 20)
    ax[2].imshow(ws_results[slice_number],cmap = cmap, interpolation='nearest')
    ax[2].axis('off')
    ax[2].set_title('Segmentation',fontsize = 20)

    fig.subplots_adjust(hspace=0.01, wspace=0.01, top=1, bottom=0, left=0,
                        right=1)
    
    return fig, ax
    
#create a function that plots the segmented image and save as image sequences

def save_seg_results(outpath, name, ws_results, cmap):
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

    Returns
    -------
    None.

    """
    
    for i in tqdm(range(len(ws_results))):
        
        fig, axes = plt.subplots(nrows=1,ncols=1)
        ax0 = axes
        ax0.imshow(ws_results[i],cmap = cmap) #choose your segmentation algorithm here
        ax0.axis('off')   #no axes so its just the image 

        fig.set_size_inches(5, 5)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        plt.savefig("{}/{}_seg_results{}.tif".format(outpath,name,i),bbox_inches='tight')
        plt.close()