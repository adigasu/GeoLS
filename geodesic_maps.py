import os
import GeodisTK
import time
import numpy as np
import SimpleITK as sitk 
from PIL import Image
from skimage.morphology import skeletonize, erosion, dilation, ball
import argparse


def geodesic_distance_3d(I, S, spacing, lamb=1, iter=4):
    '''
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)


def normalize_zero_to_one(var):
    var -= var.min()
    var /= var.max()
    return var


def background_geodesic_map(img, nb_classes=1):
    temp = np.zeros_like(img[0])
    count = 0
    for c in range(1,nb_classes):
        if img[c,...].sum() == 0:
            continue
        temp += img[c,...] 
        count +=1

    if count == 0:
        print("No foreground object!!!!")
        img[0,...] = np.ones_like(img[c,...])
    else:
        img[0,...] = 1- temp/count
    return img


def invert_geodesic_maps(img, invert_type=None, gamma=1):
    if invert_type == "exp":
        img = np.exp(-img)
    elif invert_type == "exp_gamma":
        img = np.exp(- gamma * img)
    else:
        img = img.max()-img
    img = normalize_zero_to_one(img)
    return img


def get_data(input_name, is_seg=False):

    if not os.path.isfile(input_name):
        print("File not exists:", input_name)
        return -1

    img = sitk.ReadImage(input_name)
    np_img = sitk.GetArrayFromImage(img)
    spacing_raw = img.GetSpacing()
    if is_seg:
        return np.asarray(np_img, np.uint8), spacing_raw 
    else:
        return np.asarray(np_img, np.float32), spacing_raw


def write_data4d(arr, save_path):
    img = sitk.GetImageFromArray(arr, isVector=False)
    sitk.WriteImage(img, save_path)


def mask_erosion(img, sizes=1):
    footprint = ball(sizes)
    e_img = erosion(img, footprint)
    return e_img


def mask_dilation(img, sizes=1):
    footprint = ball(sizes)
    e_img = dilation(img, footprint)
    return e_img


def get_skeleton(img):
    seed = np.zeros_like(img, np.uint8)
    seed = skeletonize(img)
    seed[seed>0] = 1
    return seed


def get_seeds(seg, nb_classes=1):
    seeds = np.zeros((nb_classes, seg.shape[0], seg.shape[1], seg.shape[2]), np.uint8)
    mask = seg.copy()
    for c in range(1, nb_classes):
        seeds[c] = get_skeleton(mask == c)

    return seeds  


def get_geodesic_distance(img_path, seg_path, nb_classes=1, dataset='FLARE', lamb=1.0):
    img, _ = get_data(img_path)
    seg, _ = get_data(seg_path, is_seg=True)

    if dataset == 'FLARE':
        spacing = [2.5, 2.0, 2.0]

    elif dataset == 'BraTS':
        spacing = [1.0, 1.0, 1.0]
        seg[seg == 4] = 3

    else:
        spacing = [1.0, 1.0, 1.0]
        print('New the dataset, define spacing or get spacing from data!!!')

    geodesic_maps = np.zeros((nb_classes, seg.shape[0], seg.shape[1], seg.shape[2]), np.float32)
    seeds = get_seeds(seg, nb_classes)

    for c in range(1, nb_classes):
        if seeds[c].sum() == 0:
            print("seeds[{}] is zero".format(c))
            continue

        gd = geodesic_distance_3d(img, seeds[c], spacing, lamb)
        geodesic_maps[c] = invert_geodesic_maps(gd, "exp_gamma", 1/gd.mean())

        if geodesic_maps[c].max() > 1.0 or geodesic_maps[c].min() < 0:
            print("geodesic_maps[c] is not [0,1]", c, geodesic_maps[c].max(), geodesic_maps[c].min())
            
        if np.isnan(geodesic_maps[c]).sum() != 0 :
            print("geodesic_maps[c] is Nan", c)
    
    geodesic_maps = background_geodesic_map(geodesic_maps, nb_classes)
    return geodesic_maps


def generate_geodesic_maps(dataset='FLARE', nb_classes=1, data_path='./FLARE21/dataset', output_path='./FLARE21/geodesic_maps'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    img_name_list = []
    for img_name in os.listdir(data_path):
        img_name_list.append(img_name)

        print("{}. Starting.. {}".format(len(img_name_list), img_name))

        if dataset == 'FLARE':
            img_path = os.path.join(data_path, img_name, img_name + "_img.nii.gz") 
            seg_path = os.path.join(data_path, img_name, img_name + "_seg.nii.gz") 

            img_gd = get_geodesic_distance(img_path, seg_path, nb_classes, dataset)

        elif dataset == 'BraTS':
            flair_path = os.path.join(data_path, img_name, img_name + "_flair.nii.gz") 
            t1ce_path = os.path.join(data_path, img_name, img_name + "_t1ce.nii.gz") 
            t1_path = os.path.join(data_path, img_name, img_name + "_t1.nii.gz") 
            t2_path = os.path.join(data_path, img_name, img_name + "_t2.nii.gz") 
            seg_path = os.path.join(data_path, img_name, img_name + "_seg.nii.gz") 

            flair_gd = get_geodesic_distance(flair_path, seg_path, nb_classes, dataset)
            t1ce_gd = get_geodesic_distance(t1ce_path, seg_path, nb_classes, dataset)
            t1_gd = get_geodesic_distance(t1_path, seg_path, nb_classes, dataset)
            t2_gd = get_geodesic_distance(t2_path, seg_path, nb_classes, dataset)

            img_gd = (flair_gd  + t1ce_gd + t1_gd + t2_gd)/4.0
        else:
            print('Add the dataset similar to FLARE')

        save_dir = os.path.join(output_path, "skeleton", img_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, img_name + "_gd.nii.gz")
        write_data4d(img_gd, save_path)
        print(" ")
    
    print("Total number of images processed: {}".format(len(img_name_list)))


if __name__ == "__main__":
    '''
    dataset: dataset name [FLARE, BraTS]
    num_classes: number of classes in the dataset
    input_dir: input data directory containing folder-wise data of volume and its mask
    output_dir: output data directory of classwise Geodesic maps
    num_seeds: number of random seed points to generate Geodesic maps (optional)
    '''
    parser = argparse.ArgumentParser(description='Geodesic Maps')
    parser.add_argument('--dataset', default='FLARE', help="options:[FLARE, BraTS]")
    parser.add_argument('--num_classes', default=5, type=int, help="number of classes")
    parser.add_argument('--input_dir', default='./FLARE21/dataset', help='input data directory')
    parser.add_argument('--output_dir', default='./FLARE21/geodesic_maps', help='output data directory')
    args = parser.parse_args()

    generate_geodesic_maps(args.dataset, args.num_classes, args.input_dir, args.output_dir)
