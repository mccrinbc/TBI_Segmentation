import nibabel as nib
import numpy as np

import argparse
import os
from glob import glob as glob
from PIL import Image
import sys

'''
This is a utilitiy script to convert your nii(.gz) files into tif. 
We specify a rotational aspect of the script and defaults to switching the volume into 1,0,2.
Our data is always received in [182,218,182], so we rearrange the slices. This does not matter
for processing purposes, but changes the axial slices to be the correct orientation. 

'''

def arg_parser():
    parser = argparse.ArgumentParser(description='Re-Arrange Dimentions of 3D')
    parser.add_argument('img_dir', type=str, 
                        help='path to nifti image directory')
    parser.add_argument('out_dir', type=str, 
                        help='path to output')
    parser.add_argument('-dims', '--dims', type=str, default='1,0,2', 
                        help='dimentions to change into')
    parser.add_argument('-split', type = int, default = 1)
    parser.add_argument('-a', '--axis', type=int, default=0, 
                        help='axis of the 3d image array on which to sample the slices')
    parser.add_argument('-p', '--pct-range', nargs=2, type=float, default=(0.2,0.8),
                        help=('range of indices, as a percentage, from which to sample ' 
                              'in each 3d image volume. used to avoid creating blank tif '
                              'images if there is substantial empty space along the ends '
                              'of the chosen axis'))
    return parser

def split_filename(filepath):
    path = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    base, ext = os.path.splitext(filename)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    return path, base, ext

def nii_to_tif(img,base,ext,args):
    if img.ndim != 3:
        print(f'Only 3D data supported. File {base}{ext} has dimension {img.ndim}. Skipping.')
    start = int(args.pct_range[0] * img.shape[args.axis])
    end = int(args.pct_range[1] * img.shape[args.axis])
    for i in range(start, end):
        I = Image.fromarray(img[i,:,:], mode='F') if args.axis == 0 else \
            Image.fromarray(img[:,i,:], mode='F') if args.axis == 1 else \
            Image.fromarray(img[:,:,i], mode='F')
        I.save(os.path.join(args.out_dir, f'{base}_{i:04}.tif'))
    return 0

def main():
    try:
        args = arg_parser().parse_args()
        filenames = glob(os.path.join(args.img_dir,'*.nii*'))
        
        if args.dims != None:
            dims = list(args.dims.split(','))
        else:
            dims = ['0','1','2'] #If we don't want to rotate, then do nothing
        
        for f in filenames:
            _, base, ext = split_filename(f)
            img = nib.load(f).get_fdata().astype(np.float32).squeeze()
            #img = np.moveaxis(img,[0,1,2],[int(dims[0]),int(dims[1]),int(dims[2])])
            
            if args.split == 1:
                nii_to_tif(img,base,ext,args)

    except Exception as e:
        print(e)
        return 1
    
if __name__ == "__main__":
    sys.exit(main())