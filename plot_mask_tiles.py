"""
Plot a selection of mask and image tiles
"""
import argparse


def main(mask_f, img_f, stop=None, start=None, step=None, show=False):
    import numpy as np
    if not show:
        # Use a headless backend in-case we don't have a DISPLAY
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mask = np.load(mask_f)['mask']
    imgs = np.load(img_f)

    for i in range(start, stop, step):
        im, m = imgs[i], mask[i]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(im)
        ax2.imshow(m[:, :, 0])
        if show:
            plt.show()
        else:
            plt.savefig("{}.png".format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('mask_f', help="Input numpy mask file", type=argparse.FileType('rb'))
    parser.add_argument('img_f', help="Input numpy file containing images", type=argparse.FileType('rb'))
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--stop', default=10, type=int)
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--show', help="Plot images to screen", action='store_true')

    # Gets command line args by default
    args = vars(parser.parse_args())

    main(**args)
