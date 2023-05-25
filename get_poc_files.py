
#!/usr/bin/env python
"""
Return a subset of filenames which contain a POC based on the supplied mask
"""
import argparse
import numpy as np

def main(input_file_list, input_mask, outfile):

    print("Reading filelist from {}".format(input_file_list))
    files = input_file_list.readlines()
    masks = np.load(input_mask)['mask']

    # Recombine the masks, note this assumes the original size
    combined_masks = masks.transpose((0,2,1,3)).reshape(-1,448,224,1).transpose((0,2,1,3))

    print(combined_masks.shape)
    print(len(files))
    assert combined_masks.shape[0] == len(files), "Mask file shape doesn't match file list"
    
    has_poc = combined_masks.any(axis=(1,2,3))
    print(has_poc.shape, has_poc.sum())
    
    subset_files = np.asarray(files)[has_poc]
    print("Writing output file...")
    outfile.writelines(subset_files)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input_file_list', help="Input file listing MODIS tiles", type=argparse.FileType('r'))
    parser.add_argument('input_mask', help="Input mask file", type=argparse.FileType('rb'))
    parser.add_argument('-o', '--outfile', help="Output name", default='poc_images.txt', type=argparse.FileType('w'))

    # Gets command line args by default
    args = vars(parser.parse_args())

    main(**args)

