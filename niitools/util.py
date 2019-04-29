import nibabel as nib

class SliceMaker(object):
    def __getitem__(self, slic):
        return slic

slicer = SliceMaker()

def binarize(infile, outfile, threshold):

    # load file
    nii = nib.load(infile)
    vol = nii.get_data()

    # create new NII
    newNii = nib.Nifti1Image((vol>=float(threshold)).astype(int), header=nii.get_header(), affine=nii.get_affine())

    # save niftis
    newNii.to_filename(outfile)

    return(newNii)