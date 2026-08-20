


# from PIL import Image, ImageDraw 
# from skimage import io
from src.imports import*


#loading images
def load(dirname, start_slice, slices): 
    flow_data = []

    fname = (os.listdir(dirname))
    for i in range(start_slice, start_slice + slices):
        im = Image.open(os.path.join(dirname, fname[i]))
        imarray = np.array(im)
        flow_data.append(imarray)

        
    # convert to a 3D array and normalise so data is between 0 and 1 
    flow_data  = np.asarray(flow_data) 
    flow_data  = preprocess(flow_data)
    return(flow_data)


def preprocess(img: np.ndarray, normalize_axis=None) -> np.ndarray:
    """Converts image to 8-bit image. Optionally normalizes along `normalize_axis`."""

    assert isinstance(img, np.ndarray)
    assert img.dtype in (np.uint8, np.uint16, np.float32, np.float64), f"Unsupported dtype: {img.dtype}"
    assert normalize_axis is None or 0 <= normalize_axis % img.ndim < img.ndim

    if img.dtype in (np.float32, np.float64):
        img = (img - img.min()) / (img.max() - img.min()) * 255
        #mn = np.min(img, axis=normalize_axis, keepdims=True)
        #mx = np.max(img, axis=normalize_axis, keepdims=True)
        #np.subtract(img, mn, out=img)  # img = img - img.min()
        #np.divide(img, mx - mn, out=img)
        #np.multiply(img, 255, out=img)
        img = img.astype(np.uint8)
    
    if img.dtype == np.uint16:
        img = (img - img.min()) / (img.max() - img.min()) * 255
        #mn = np.min(img, axis=normalize_axis, keepdims=True)
        #mx = np.max(img, axis=normalize_axis, keepdims=True)
        #np.subtract(img, mn, out=img)  # img = img - img.min()
        #np.divide(img, mx - mn, out=img)
        #np.multiply(img, 255, out=img)
        img = img.astype(np.uint8)
    
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    
    return img


# # function to mask with the dry scan 
# def mask_with_dry(img, dry_scan):
#     if img.max() > 1:
#         img = (img - img.min()) / (img.max() - img.min()) * 255
#         img = img.astype(np.uint8)

#     # creating mask from segmented dry scan
#     mask = (dry_scan == 0)

#     assert img.shape == mask.shape
#     assert mask.dtype == np.bool8
#     # mask image 
#     foreground = img.copy()
#     foreground[mask] = 255 

#     # create a composite image using the alpha layer
#     masked_img = np.array(foreground, dtype=np.uint8)
#     return masked_img 


def sanity_check(img: np.ndarray, mask: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.bool8
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert len(mask.shape) == 2
    assert len(img.shape) == 2  # grayscale

    background = einops.repeat(img, "h w -> h w c", c=3)  # grayscale to rgb

    foreground = background.copy()
    foreground[mask] = [255, 0, 0]

    foreground = foreground.astype(np.float16)
    background = background.astype(np.float16)

    composite = background * (1.0 - alpha) + foreground * alpha
    composite = np.array(composite, dtype=np.uint8)

    assert isinstance(composite, np.ndarray)
    assert composite.dtype == np.uint8
    assert len(composite.shape) == 3  # rgb

    return composite


def simple_thresholding(img: np.array, min_threshold: float, max_threshold: float) -> np.array:
    return ((img.max() - img.min()) * min_threshold + img.min() <= img) & (img <= (img.max() - img.min()) * max_threshold + img.min())



def load_image(path: str, as_gray: bool = True):
    """
    Loads an image, an image stack (3D) or numpy array from the given path.

    Parameters
    ----------
    path : str
        The path to the image file or the directory containing a sequence of image slices.
    as_gray : bool, optional
        If True, the image(s) are converted to grayscale. Default is False.

    Returns
    -------
    image_array : np.ndarray
        The loaded image data. If a directory is given, returns a 3D NumPy array where each slice
        along the third axis corresponds to a 2D image from the sequence.
    """
    
    if os.path.isdir(path):
        # Case 1: Loading a 3D image stack from a directory of image slices
        # List all image files in the directory and sort them
        file_list = sorted([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        
        if not file_list:
            raise FileNotFoundError("No image files with png, jpg, jpeg, tif, tiff format found in the specified directory.")
        
        # Load each image and store it in a list
        images = [io.imread(os.path.join(path, f), as_gray = as_gray) for f in file_list]
        # Stack images along the third dimension (depth)
        image_array = np.stack(images, axis=-1)
    
    
    elif path.lower().endswith('.npy'):
    # Case 2: Loading a NumPy array from a .npy file
        # Load the NumPy array
        image_array = np.load(path)
    
    
    else:
        ## Case 3: Loading a single image from a file path
        if not os.path.exists(path):
            raise FileNotFoundError(f"The file in {path} | does not exist.")
        
        # Load the image
        image_array = io.imread(path, as_gray= as_gray)

    return image_array




def nlm_denoise(Img,patch,dist):
    
    def nlm_denoise_wrap(array):
        correct=img_as_float(array[0])
        sigma_est = np.mean(estimate_sigma(correct))
        correct = denoise_nl_means(correct, h=1.2 * sigma_est, sigma=sigma_est,fast_mode=True,patch_size=patch, patch_distance=dist)
        return correct[np.newaxis, ...]
    
    denoise=skimage.util.apply_parallel(nlm_denoise_wrap,Img, chunks=(1, Img.shape[1],Img.shape[2]), dtype='float',compute=True)
    
    return  denoise




def QC_filter(denoise2_fast, dry_data):

    #note we want the most filtered image that doesn't compromise the boundaries. 
    diff_img = denoise2_fast- dry_data

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 20),
                        sharex=True, sharey=True)

    ax[0].imshow(diff_img[15,:, :],cmap='gray')
    ax[0].axis('off')


    ax[1].imshow(dry_data[15,:, :],cmap='gray')
    ax[1].axis('off')

    ax[2].imshow(denoise2_fast[15,: :],cmap='gray')
    ax[2].axis('off')

    fig.tight_layout()
    plt.show()

    #plot of the histograms (entire image, not the zoomed in image)
    bins=255
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize =(7, 5))
    ax.set_xlabel('Greyscale value')
    ax.set_ylabel('Count')

    ax.set_xlim(0, 120) 
    ax.set_ylim(0, 0.1*10**7) 

    ax.hist(dry_data.ravel(), bins=bins, histtype='step', color='black')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    ax.hist(preprocess(denoise2_fast).ravel(), bins=bins, histtype='step', color='red')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.legend(['raw data', 'filtered data'])
    fig.tight_layout()
    plt.show()



def phase_segmentation_adapted_watershed(im, pore_thresh, bins = 255):
    #initial 2D code stolen from Griffin Chure

    """
    Performs watershed segmentation on the greyscale image 
    
    Parameters
    ----------
    im : 3D array
        Image to be segmented. 
    
    grain_thresh: float
        Threshold above which grains exist. 
    pore_thresh: float 
        Threshold below which pores exist. 
    
    Output
    -------
    final_seg: segmented image 
    """
    
    # Make sure that the input image is between 0 and 1 
    if np.max(im) != 1.0:
        im = (im - im.min()) / (im.max() - im.min())

    im = im.astype(np.float32)
    #plot histogram with limits of what is pore and what is grain shown
    fig, ax = plt.subplots(figsize =(10, 7))
    hist_values = im.ravel()
    hist_values = hist_values[hist_values < 0.99]
    y, x, _ = plt.hist(hist_values, bins=bins, histtype='step', color='red')


    # Normal distribution to get the max threshold 
    loc         =  np.argmax(y)                             #find peak 
    update_list = hist_values[hist_values > x[loc]]
    update_y, update_x, _ = plt.hist(update_list, bins=bins, histtype='step', color='red')
    loc2                  =  np.argmax(update_y[update_y< (0.0001 * y.max())])

    grain_thresh = (x[loc] - (update_x[loc2] - x[loc])) *0.85 #currently a 15% leeway 
    plt.axvline(grain_thresh, color='r', linestyle='dashed', linewidth=1)
    plt.axvline(pore_thresh, color='k', linestyle='dashed', linewidth=1)
    plt.show()

    # Generate the catchment basins for watershed
    print('Making the catchment basin')
    basins = np.zeros_like(im)
    basins[im < grain_thresh] = 1
    basins[im > pore_thresh] = 2
    basins = basins.astype(np.int32)

    # Peform the watershed by flooding. 
    print('Flood the basin')
    flood_seg = skimage.segmentation.watershed(im , basins)
    flood_seg = flood_seg > 1.0

    # Compute the distance matrix
    print('Compute the distance matrix')
    distances = ndi.distance_transform_edt(flood_seg)
    
    #Find the maxima NB this is taking a while on 3D images 
    print('Find the maxima')
    local_max = skimage.morphology.local_maxima(distances)
    max_lab = skimage.measure.label(local_max)

    #Perform the topological watershed. 
    print('Perform the topological watershed')
    final_seg = skimage.segmentation.watershed(-distances, max_lab, mask=flood_seg)

    # Remove any stray crap. 
    final_seg = skimage.morphology.remove_small_objects(final_seg, min_size=4)

    #Normalise final_seg 
    final_seg = (final_seg - final_seg.min()) / (final_seg.max() - final_seg.min()) * 255
    # Subresolvable pores - large regions in the middle zone 
    sub_resolvable = simple_thresholding(im, grain_thresh*0.95, pore_thresh*1.15)
    sub_resolvable = skimage.morphology.remove_small_objects(sub_resolvable, min_size=10)   
    print('Grain threshold is', grain_thresh)
    return final_seg, sub_resolvable








def multiple_runs(dry_float, patch, dist, pore_thresh):
    #Iterating over different filtering values 
    dry_float = dry_float.astype(np.float32)
    denoise2_fast = nlm_denoise(dry_float, patch = patch,  dist = dist)
    QC_filter(denoise2_fast, dry_float)
    input_segmentation                   = img_as_ubyte(preprocess(denoise2_fast.transpose(1,2,0))) #must transpose the image for the input to be xyz, previously zxy
    segmented_img, sub_resolved_pores    = phase_segmentation_adapted_watershed(input_segmentation, pore_thresh=pore_thresh)
    
    #convert to binary and check the segmentation 
    segmented_binary             = np.where(segmented_img>0, 1, segmented_img).astype(np.bool_)
    greyscale_input              = (input_segmentation - input_segmentation.min()) / (input_segmentation.max() - input_segmentation.min()) * 255


    # seg_dry = sanity_check(greyscale_input[:,:,3].astype(np.uint8), segmented_binary[:,:,3]==0, 0.2)
    #plot of the histograms (entire image, not the zoomed in image)
    bins=255
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize =(7, 5))
    ax.set_xlabel('Greyscale value')
    ax.set_ylabel('Count')

    ax.set_xlim(0, 120) 
    ax.set_ylim(0, 0.1*10**7) 

    ax.hist(dry_float.ravel(), bins=bins, histtype='step', color='black')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    ax.hist(preprocess(denoise2_fast).ravel(), bins=bins, histtype='step', color='red')
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax.legend(['raw data', 'filtered data'])
    fig.tight_layout()
    plt.show()


    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20),
                        sharex=True, sharey=True)

    ax[0].imshow(segmented_binary[:,:,3],cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('Segmented pore space')


    ax[1].imshow(greyscale_input[:,:,3],cmap='gray')
    ax[1].axis('off')
    ax[1].set_title('original image')

    fig.tight_layout()
    plt.show()
    porosity = np.sum(segmented_binary==0)/ np.sum(segmented_binary==1)
    return porosity, segmented_binary


def numpy_to_raw(arr, path, dtype=np.uint8):
    arr = np.asarray(arr, dtype=dtype)
    arr.tofile(path)








