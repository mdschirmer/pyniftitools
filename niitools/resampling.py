#!/data/porpoise/Imaging/Packages/virtualenvs/WMH/bin/python
from __future__ import print_function
from __future__ import division
import sys
import ast

import numpy as np
from numpy import newaxis as nax
import nibabel as nib

import scipy.ndimage as sn
import scipy.ndimage.interpolation as sni
import pickle
import gc

###
# ToDo: 
#  - Fix affine transformation of individual axis resampling (see e.g. upsample())
###

def downsample_axis(infile, outfile, axis, new_pixdim, method='linear'):
    """
    Downsamples a volume along a specified axis.

    Inputs
    ------
    infile : a filename from which to read data
    outfile : a filename to which to save data
    axis : the axis along which to downsample
    pixdim_ratio : the ratio by which to decrease pixdim.
    method : interpolation method ('linear' or 'nearest')
    """
    if type(new_pixdim) is str:
        new_pixdim = ast.literal_eval(new_pixdim)
    if type(axis) is str:
        axis = ast.literal_eval(axis)
    from scipy.interpolate import interpn
    nii = nib.load(infile)
    hdr = nii.get_header()
    aff = nii.get_affine()
    data = nii.get_data().astype('float32')

    in_coords = []
    out_coords = []
    affine_modifier = np.eye(3)
    for ax in [0,1,2]:
        in_coords.append(np.arange(256))
        if ax == axis:
            out_slice = slice(0, 252, new_pixdim)
            affine_modifier[ax,ax] = new_pixdim
        else:
            out_slice = slice(0, 256)
        out_coords.append(out_slice)

    out_grid = np.mgrid[out_coords].transpose(1,2,3,0)

    new_data = interpn(in_coords, data, out_grid, method=method, fill_value=None)
    hdr['pixdim'][1+axis] = new_pixdim
    # Multiply affine matrix by resampling matrix.
    # WARNING: no guarantees this'll work for non-axis-aligned images...
    aff[:3, :3] = np.dot(affine_modifier, aff[:3, :3])
    #new_aff = np.vstack((np.dot(affine_modifier, aff[:-1,:]), aff[-1:,:]))

    out = nib.Nifti1Image(new_data.astype('uint8'), header=hdr.copy(), affine=aff)
    out.update_header()
    out.to_filename(outfile)

def upsample_axis(infile, outfile, outmask, axis, pixdim_ratio, method='linear'):
    """
    Upsamples a volume along a specified axis.

    Inputs
    ------
    infile : a filename from which to read data
    outfile : a filename to which to save data
    outmask : a filename to which to save the upsampling mask
    axis : the axis along which to upsample
    pixdim_ratio : the ratio by which to increase pixdim.
                   if integer, inserts (#-1) slices between each
    method : interpolation method ('linear' or 'nearest')
    """
    if type(pixdim_ratio) is str:
        pixdim_ratio = ast.literal_eval(pixdim_ratio)
    if type(axis) is str:
        axis = ast.literal_eval(axis)
    from scipy.interpolate import interpn
    nii = nib.load(infile)
    hdr = nii.get_header()
    aff = nii.get_affine()
    data = nii.get_data().astype('float32')

    in_coords = []
    out_coords = []
    affine_modifier = np.eye(3)
    mask_coords = []
    for ax in [0,1,2]:
        pixdim = hdr['pixdim'][1+ax]
        # cheat/hack
        if abs(pixdim - round(pixdim)) < .05:
            pixdim = round(pixdim)
        n_slices = data.shape[ax]

        cap = pixdim * (n_slices-1) + (.01*pixdim)

        slicer = np.arange(0, cap, pixdim)
        in_coords.append(slicer)
        if ax == axis:
            out_slice = slice(0, cap, pixdim / pixdim_ratio)
            affine_modifier[ax,ax] = 1/pixdim_ratio
            mask_slice = slice(0, n_slices*pixdim_ratio, pixdim_ratio)
        else:
            out_slice = slice(0, cap, pixdim)
            mask_slice = slice(0, n_slices)
        out_coords.append(out_slice)
        mask_coords.append(mask_slice)

    out_grid = np.mgrid[out_coords].transpose(1,2,3,0)
    slices = [slice(None) for i in [0,1,2]] + [axis]
    slices[axis] = -1
    assert np.allclose(out_grid[slices], out_grid[slices].max())

    # Hack to avoid numerical issues where the output coordinates
    # might be very slightly beyond the range due to floating point
    # error
    if np.allclose(out_grid[slices].max(), in_coords[axis][-1]):
        out_grid[slices] = in_coords[axis][-1]

    new_data = interpn(in_coords, data, out_grid, method=method, fill_value=None)
    mask = np.zeros_like(new_data)
    mask[mask_coords] = 1
    hdr['pixdim'][1+axis] = pixdim / pixdim_ratio
    # Multiply affine matrix by resampling matrix.
    # WARNING: no guarantees this'll work for non-axis-aligned images...
    aff[:3, :3] = np.dot(affine_modifier, aff[:3, :3])
    #new_aff = np.vstack((np.dot(affine_modifier, aff[:-1,:]), aff[-1:,:]))

    out = nib.Nifti1Image(new_data, header=hdr.copy(), affine=aff)
    out.update_header()
    out.to_filename(outfile)

    out_mask = nib.Nifti1Image(mask, header=hdr.copy(), affine=aff)
    out_mask.update_header()
    out_mask.to_filename(outmask)


## for all axes at the same time

def upsample(niiFileName, upsampledFile, zoom_values_file='upsampling_log.pickle', isotrop_res=True, upsample_factor=None, polynomial='3'):
    """
    Upsample a nifti and save the upsampled image.
    The upsampling procedure has been implemented in a way that it is easily revertable.

    Example arguments:
    niiFileName = '10529_t1.nii.gz'
    upsampled_file = '10529_t1_upsampled.nii.gz'
    """

    # load the nifti
    nii = nib.load(niiFileName)
    header = nii.get_header()
    affine = nii.get_affine()

    # make data type of image to float 
    out_dtype = np.float32
    header['datatype'] = 16 # corresponds to float32
    header['bitpix'] = 32 # corresponds to float32

    # in case nothing should be done
    isotrop_res = bool(int(isotrop_res))
    if ((not isotrop_res) and (upsample_factor is None)):
        print('Uspampling not requested. Skipping...')
        nii.to_filename(upsampledFile)
        return nii

    # convert input to number
    if isotrop_res:
        isotrop_res = float(np.min(header.get_zooms()[0:3]))
        all_upsampling = [float(zoom)/isotrop_res for zoom in header.get_zooms()[0:3]]
        for idx, zoom in enumerate(all_upsampling):
            if zoom<1:
                all_upsampling[idx]= 1.
            else:
                all_upsampling[idx]= np.round(zoom)
    else:
        upsample_factor = float(upsample_factor)
        all_upsampling = [upsample_factor for zoom in header.get_zooms()[0:3]]
        

    polynomial = int(polynomial)

    old_volume = np.squeeze(nii.get_data().astype(float))
    # get new volume shape and update upsampling based on rounded numbers
    old_shape = old_volume.shape
    print(old_shape)
    new_shape = tuple([np.round(old_shape[ii]*usampling).astype(int) for ii, usampling in enumerate(all_upsampling)])
    print(new_shape)
    # all_upsampling = [float(new_shape[ii])/float(old_shape[ii]) for ii in np.arange(len(old_shape))]
    # print('Upsampling with scales: ' + str(all_upsampling))

    # # get indices
    # a,b,c = np.indices(new_shape)
    # a = a.ravel()
    # b = b.ravel()
    # c = c.ravel()

    # upsample image
    print('Upsampling volume...')
    # u_values = sni.map_coordinates(old_volume, [a/all_upsampling[0], b/all_upsampling[1], c/all_upsampling[2]])
    # del nii
    # gc.collect()
    # vol = np.zeros(new_shape, dtype = out_dtype)

    # for jj in np.arange(len(a)):
    #     vol[a[jj], b[jj], c[jj]] = u_values[jj].astype(out_dtype)

    # # vol[vol<=0.] = 0 # nonsensical values

    vol = sn.zoom(old_volume, all_upsampling)

    print('Done.')

    # update voxel sizes in header
    if len(header.get_zooms())==3:
        new_zooms = tuple( [header.get_zooms()[ii]/float(all_upsampling[ii]) for ii in np.arange(3)] ) # 3 spatial dimensions
    elif len(header.get_zooms())>3:
        tmp = [header.get_zooms()[ii]/float(all_upsampling[ii]) for ii in np.arange(3)]
        tmp.extend(list(header.get_zooms()[3:]))
        new_zooms = tuple(tmp) # 3 spatial dimensions + 1 time
    else:
        print('Cannot handle this stuff... ')
        print(header.get_zooms())
        raise Exception('Header has less than 2 entries. 2D?')

    header.set_zooms(new_zooms)

    # adapt affine according to scaling
    all_upsampling.extend([1.]) # time
    scaling = np.diag(1./np.asarray(all_upsampling))
    affine = np.dot(affine, scaling)

    # create new NII
    newNii = nib.Nifti1Image(vol.astype(out_dtype), header=header, affine=affine)

    # save niftis
    newNii.to_filename(upsampledFile)

    # save upsampling factors
    a=0
    b=0
    c=0
    with open(zoom_values_file, 'w') as outfile:
        pickle.dump([np.unique(a),np.unique(b),np.unique(c),all_upsampling[:-1],polynomial, old_shape], outfile)


    return (newNii)

def downsample(niiFileName, downsampled_file, zoom_values_file='upsampling_log.pickle', order=3):
    """
    downsample a nifti which has been upsampled with the function above.

    Example arguments:
    niiFileName = '10529_t1_upsampled.nii.gz'
    downsample_file = '10529_t1_downsample.nii.gz'
    zoom_values_file = 'upsampling_log.pickle'
    """

    # load the nifti
    nii = nib.load(niiFileName)
    header = nii.get_header()

    # make data type of image to float 
    out_dtype = np.float32
    header['datatype'] = 16 # corresponds to float32
    header['bitpix'] = 32 # corresponds to float32
    
    downsample_factor=[]
    with open(zoom_values_file,'r') as zfile:
        [a, b, c, all_upsampling, polynomial, old_shape] = pickle.load(zfile)

    print('Downsampling with scales: ' + str(1./np.asarray(all_upsampling)))
    if old_shape:
        current_shape = nii.get_data().shape
        print(old_shape)
        downsample_values = np.asarray([float(old_shape[ii])/float(current_shape[ii]) for ii in np.arange(len(old_shape))])
    else:
        if len(all_upsampling) == 1:
            downsample_values = 1./np.asarray(3*all_upsampling)
        else:
            downsample_values = 1./np.asarray(all_upsampling)

    # #prepping for loop
    # all_coords = [a,b,c]
    # # print(np.arange(np.round(np.max(coords))))

    # downsampling image
    print('Downsampling volume...')
    # downsample_indices = []
    # for idx, factor in enumerate(downsample_values):
    #     print('%f, %f'%(idx, factor))
    #     coords = (all_coords[idx]*factor)
    #     downsample_idx = []
    #     for jj in np.arange(np.round(np.max(coords))):
    #         downsample_idx.append(np.where(coords==jj)[0])
        
    #     downsample_indices.append(downsample_idx)

    # # TODO: change this to use slices notation
    # vol = nii.get_data().astype(out_dtype)[np.squeeze(np.asarray(downsample_indices[0])),:,:]
    # vol = vol[:,np.squeeze(np.asarray(downsample_indices[1])),:]
    # vol = vol[:,:,np.squeeze(np.asarray(downsample_indices[2]))]
    # if np.sum(np.abs([vol.shape[ii] - old_shape[ii] for ii in np.arange(len(old_shape))])) != 0:
    #     print('Downsampled output shape not the same as target.')
    vol = sn.zoom(nii.get_data(), downsample_values, order=int(order))
    print('Done.')

    # update voxel sizes in header
    if len(header.get_zooms())==3:
        new_zooms = tuple( [header.get_zooms()[ii]/float(downsample_values[ii]) for ii in np.arange(3)] ) # 3 spatial dimensions
    elif len(header.get_zooms())>3:
        tmp = [header.get_zooms()[ii]/float(downsample_values[ii]) for ii in np.arange(3)]
        tmp.extend(list(header.get_zooms()[3:]))
        new_zooms = tuple(tmp) # 3 spatial dimensions + 1 time
    else:
        print('Cannot handle this stuff... ')
        print(header.get_zooms())
        raise Exception('Header has less than 2 entries. 2D?')

    # new_zooms = tuple( [header.get_zooms()[ii]/float(downsample_values[ii]) for ii in np.arange(len(header.get_zooms()))] )
    header.set_zooms(new_zooms)

    # adapt affine according to scaling
    affine = nii.get_affine()
    downsample_values = downsample_values.tolist()
    downsample_values.extend([1.]) # time
    scaling = np.diag(1./np.asarray(downsample_values))
    affine = np.dot(affine, scaling)

    # create new NII
    newNii = nib.Nifti1Image(vol.astype(out_dtype), header=header, affine=affine)

    # save niftis
    newNii.to_filename(downsampled_file)

    return (newNii)
