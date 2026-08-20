from src.imports import*

# =============================================================================
# load the image
# =============================================================================
# dirname     = 'data'

## this function can load a stack of tif | np | etc. if we have tif it returns normlize foalt64 with value between 1 and 0
## Data type: float64 in range [0, 1] (because scikit‑image converts to grayscale and normalises)
# dry_data = utils.load_image(dirname, as_gray = True) 

## loading one 3D tif
path = 'data/dry_bin2.tif'
dry_data  = skimage.io.imread(path)
dry_float = dry_data.astype(np.float32)
## apply non loc mean
denoise2_fast = utils.nlm_denoise(dry_data.astype(np.float32), patch = 3,  dist = 4)
utils.QC_filter(denoise2_fast, dry_data)

## apply segmentation
bins = 255
input_segmentation                   = img_as_ubyte(utils.preprocess(denoise2_fast.transpose(1,2,0))) # must transpose the image for the input to be xyz, previously zxy
segmented_img, sub_resolved_pores    = utils.phase_segmentation_adapted_watershed(input_segmentation, pore_thresh=0.21, bins = 255)


#convert to binary and check the segmentation 
segmented_binary             = np.where(segmented_img>0, 1, segmented_img).astype(np.bool_)
segmented_binary_subpore     = np.where(sub_resolved_pores>0, 1, sub_resolved_pores).astype(np.bool_)
greyscale_input              = (input_segmentation - input_segmentation.min()) / (input_segmentation.max() - input_segmentation.min()) * 255

porosity = np.sum(segmented_binary==0)/ np.sum(segmented_binary==1)

plt.imshow(greyscale_input[0, :, :], 'gray')
plt.show()
plt.imshow(segmented_binary[0, :, :], 'gray')
plt.show()
plt.imshow(segmented_binary_subpore[0, :, :], 'gray')
plt.show()

# =============================================================================
# saving
# =============================================================================

path = 'result/img_save_tst.raw'
utils.numpy_to_raw(segmented_binary, path, dtype=np.uint8)


# =============================================================================
# functions for sensitivity analysis | pore tresh & filter patch
# =============================================================================

def single_run(dry_data, pore_thresh, patch=3, dist=4 ):

    porosity, segmented_binary =  utils.multiple_runs(dry_data, patch, dist, pore_thresh)
    
    return porosity, segmented_binary


# porosity, segmented_binary = single_run(dry_data, 0.21)


def pore_tresh_sa(dry_data, linspace):
    phi_pore_thresh = []

    for i in np.linspace(linspace[0], linspace[1], linspace[2]):
        porosity, segmented_binary = utils.multiple_runs(dry_data, patch = 3, dist = 4, pore_thresh=i)
        phi_pore_thresh.append(porosity)

    plt.plot(phi_pore_thresh, 'x')
    plt.ylabel('Porosity')
    plt.xlabel('Patch size')
    
    return phi_pore_thresh


# linspace = [0.15, 0.3, 3]  ## [a, b, c]  |   c number of run | a start | b end

# phi_pore_thresh = pore_tresh_sa(dry_data, linspace)



def filter_sa(dry_data, pore_thresh, idx):
    phi = []
    
    for i in range(idx):
        print(i)
        porosity, segmented_binary =  utils.multiple_runs(dry_data, patch = i, dist = 6, pore_thresh=pore_thresh)
        phi.append(porosity)
    
    plt.plot(phi, 'x')
    plt.ylabel('Porosity')
    plt.xlabel('Patch size')
    
    return phi

# phi = filter_sa(dry_data, 0.21, 3)



