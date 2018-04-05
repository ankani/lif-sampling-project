#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:30:53 2018

@author: achattoraj and/or richard
"""

import os
import h5py as h5
import numpy as np

images_dataset = "/Users/achattoraj/Desktop/Projects/LIF_Sampling_Project/NewModel/vanhateren_iml/"
images = [name for name in os.listdir(images_dataset) if name[-4:] == ".iml"]
(im_height, im_width) = (1024, 1536)
im_dtype = 'uint16'

n_patches = 10
patch_size = (8, 8)
destination_name = "patches_%dx%d_%d" % (patch_size + (n_patches,))
destination_file = h5.File(destination_name + ".tmp.h5", "w")

idxs = np.zeros(shape=(n_patches, 3), dtype='int32')
idxs[:, 0] = np.random.randint(len(images), size=(n_patches,)) # which image
idxs[:, 1] = np.random.randint(im_height - patch_size[0], size=(n_patches,)) # bottom edge
idxs[:, 2] = np.random.randint(im_width - patch_size[1], size=(n_patches,)) # bottom edge

# Sort idxs so that same images are adjacent
idxs = idxs[idxs[:, 0].argsort()]

def read_image(iml_filename):
    with open(iml_filename, "rb") as f:
        data = np.fromfile(f, dtype=im_dtype)
    data.byteswap(True)
    return data.reshape((im_height, im_width)).astype('float32') / np.iinfo(im_dtype).max

try:
    patches_array = destination_file.require_dataset("patches", (n_patches,) + patch_size, dtype='f')
        
    last_im_idx = -1
    for i in range(n_patches):
        im_idx = idxs[i, 0]
        
        if im_idx != last_im_idx:
            last_im_idx = im_idx
            image = read_image(os.path.join(images_dataset, images[im_idx]))
        
        bottom, left = idxs[i, 1:]
        patches_array[i, :, :] = image[bottom:bottom+patch_size[0], left:left+patch_size[1]]
    
    destination_file.close()
    
    os.rename(destination_name + ".tmp.h5", destination_name + ".h5")
except Exception as e:
    os.remove(destination_name + ".tmp.h5")
    raise(e)
