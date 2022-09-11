# -*- coding: utf-8 -*-
"""
This script reads MRI pictures in NifTI format and preprocesses images:
- normalize images
- crop to (220, 220, 88)
- reshape
Then, data are saved to .npy files (training.npy, training_labels.npy, test.npy, test_labels.npy)

"""

import os
import nibabel as nib
import numpy as np

def normalize(image):
    '''
    Normalize MRI image in each of the z-dimensions (depth).
    
    Args:
        image (np.array): input image, shape(dim_x, dim_y, dim_z)
    
    Returns:
        normalized_image (np.array): normalized version of input image
    '''
    normalized_image = image / 255
    return normalized_image


def cropping(image, new_size):
    """
    Crops image around the center.
    
    Args:
        image (tensor): image to crop.
        new_size (triple): desired size of the cropped image (H, W, C)
    """
    start_idx = ((np.array(image.shape) - np.array(new_size)) / 2).astype(int)
    end_idx = (np.array(start_idx) + np.array(new_size)).astype(int)
    cropped_im = image[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]]
    return cropped_im

def reshape(image):
    """
    Reshape image from (H, W, D) format to (C, H, W).
                        (0, 1, 2)           (2, 0, 1)
    
    Args:
        image (tensor): image to crop.
    """
    im = image[None,:,:,:]
    im = np.moveaxis(im, [0,1,2,3], [1,2,3,0])
    return im

def preprocessing(dataset_path, dataset_name, im_size):
    '''
    Read MRI images from the files and preprocess. Later save preprocessed dataset.
    
    Args:
        dataset_path (string): directory path to the dataset we want to preprocess
    '''    
    image_dataset = None
    label_dataset = None
    
    image_nifty_file = 'lgemri.nii'
    label_nifty_file = 'laendo.nii'
    
    print('[INFO] Loading NifTI files and converting to numpy arrays ...')
        
    folds_list = os.listdir(dataset_path)
    for f, fold in enumerate(folds_list):
        print(f'     sample {f + 1} / {len(folds_list)}')
        
        # image
        image_dir = dataset_path + '/' + fold + '/' + image_nifty_file
        image = np.array(nib.load(image_dir).get_fdata())
        #print(f'            image: {image.shape}')
        image = normalize(image)
        image = cropping(image, im_size)
        image = reshape(image)
        #print(f'            im: {image.shape}')
        if f == 0:
            image_dataset = image
        else:
            image_dataset = np.concatenate((image_dataset, image), axis=0)
        
        # label
        label_dir = dataset_path + '/' + fold + '/' + label_nifty_file
        label = np.array(nib.load(label_dir).get_fdata())
        #print(f'            label: {label.shape}')
        label = normalize(label)
        label = cropping(label, im_size)
        label = reshape(label)
        #print(f'            lb: {label.shape}')
        if f == 0:
            label_dataset = label
        else:
            label_dataset = np.concatenate((label_dataset, label), axis=0)
    
    print(f'     image dataset size: {image_dataset.shape}')
    print(f'     label dataset size: {label_dataset.shape}')
    
    print('[INFO] Saving datasets...')
    np.save(f'./datasets/{dataset_name}.npy', image_dataset)
    np.save(f'./datasets/{dataset_name}_labels.npy', label_dataset) 

    
if __name__ == "__main__":
    im_size = (220,220,88)
    
    print('[INFO] Start preprocessing training set...')
    training_path = './raw_dataset/2018 Atrial Segmentation Challenge/Training Set'
    preprocessing(training_path, 'training', im_size)
    
    print('[INFO] Start preprocessing test set...')
    test_path = './raw_dataset/2018 Atrial Segmentation Challenge/Testing Set'
    preprocessing(test_path, 'test', im_size)
