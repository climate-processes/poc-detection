import numpy as np
import os
import argparse
from refined_model.code.model_def import get_model as get_refined_model
from rough_model.code.model_def import get_model as get_rough_model

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7,8"


def get_masked_dataset(mask, dataset):
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))

    dilated_mask = np.zeros(mask.shape)
    for i in range(mask.shape[0]):
        dilated_mask[i, ..., 0] = cv2.dilate(mask[i, ...], kernel, iterations=6)

    dataset_masked = np.zeros(dataset.shape)
    for i in range(3):
        dataset_masked[..., i] = (dataset[..., i] + 1) / 2 * dilated_mask[..., 0]

    return dataset_masked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help="Input numpy stack of images")
    parser.add_argument('weights', help="Path to pre-trained model weights")
    parser.add_argument('-o', '--output', help="Output numpy mask stack file name", default='poc_mask.npz')
    parser.add_argument('-b', '--batch_size', help="Batch size", default=1)
    parser.add_argument('-n', '--n_gpus', help="Number of GPUs to use", defaut=1)
    parser.add_argument('-e', '--n_epochs', help="Number of epochs to run", defaut=1)
    parser.add_argument('-l', '--learning_rate', help="(Starting) Learning rate ", defaut=0.01)
    parser.add_argument('-d', '--decay_rate', help="Decay rate", defaut=0.0)

    # Gets command line args by default
    args = parser.parse_args()

    importdataset = np.memmap(args.input)
    dataset = 2 * (importdataset / 255.) - 1

    print('Input dataset shape = ', dataset.shape)

    rough_mask_model = get_rough_model(args.weights + 'rough_model_weights.h5', args.learning_rate,
                                       args.decay_rate, args.n_gpus)

    refined_mask_model = get_refined_model(args.weights+'refine_model_weights.h5', args.learning_rate,
                                           args.decay_rate, args.n_gpus)

    # create the rough masks
    mask = rough_mask_model.predict(dataset, batch_size=4*args.n_gpus)

    # How many rough POCs?
    print("Roughly detected POCs: {}".format(mask.any(axis=0).sum()))
    # TODO I might be able to speed up the following by only applying it to this subset of images

    # dilate them and apply to the dataset
    dataset_masked = get_masked_dataset(mask, dataset)

    # refine the masked dataset
    refined_mask = refined_mask_model.predict(dataset_masked, batch_size=4*args.n_gpus)

    # Create the final mask
    pre = refined_mask >= 0.5

    # How many rough POCs?
    print("Final detected POCs: {}".format(mask.any(axis=0).sum()))

    pre.savez_compressed(args.output)
    print("Done.")
