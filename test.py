import argparse
from pathlib import Path
import numpy as np
import scipy.io as scio
from sklearn.metrics import confusion_matrix

from mscapsnet.models import MSCapsNet
from mscapsnet.prepare import readdata


def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix)**2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA


def test(model, data):
    x_test, y_test = data[0], data[1]
    n_samples = y_test.shape[0]
    add_samples = args.batch_size - n_samples % args.batch_size
    x_test = np.concatenate((x_test[0:add_samples, :, :, :], x_test), axis=0)
    y_test = np.concatenate((y_test[0:add_samples, :], y_test), axis=0)
    y_pred = model.predict(x_test, batch_size=args.batch_size)
    ypred = np.argmax(y_pred, 1)
    y = np.argmax(y_test, 1)
    matrix = confusion_matrix(y[add_samples:], ypred[add_samples:])
    return matrix, ypred, add_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_class',
                        help="Number of classes",
                        default=2,
                        type=int)
    parser.add_argument('--num_routing',
                        default=3,
                        help="Should be >= 1",
                        type=int)
    parser.add_argument('--windowsize', help="Patch size", default=9, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--image_file', default='./data/YellowRiverI.mat')
    parser.add_argument('--label_file', default="./data/YellowRiverI_gt.mat")
    args = parser.parse_args()

    # Load model
    model = MSCapsNet(input_shape=[args.windowsize, args.windowsize, 3],
                      n_class=args.n_class,
                      num_routing=args.num_routing,
                      batch_size=args.batch_size)

    # Load weights
    args.save_dir = Path(args.save_dir)
    model.load_weights(args.save_dir / 'weights-test.h5')

    i = 0
    test_nsamples = 0
    RESULT = []
    matrix = np.zeros([args.n_class, args.n_class], dtype=np.float32)
    while True:
        data = readdata(args.image_file,
                        args.label_file,
                        train_nsamples=1000,
                        validation_nsamples=1000,
                        windowsize=args.windowsize,
                        istraining=False,
                        shuffle_number=np.load(
                            str(args.save_dir / 'test_shuffle_number.npy')),
                        times=i)
        if data == None:
            OA, AA_mean, Kappa, AA = cal_results(matrix)
            print('-' * 50)
            print('OA:', OA)
            print('AA:', AA_mean)
            print('Kappa:', Kappa)
            print('Classwise_acc:', AA)
            break
        test_nsamples += data[0].shape[0]
        matrix1, ypred, add_samples = test(model=model,
                                           data=(data[0], data[1]))
        matrix = matrix1 + matrix
        RESULT = np.concatenate((RESULT, ypred[add_samples:]), axis=0)
        i = i + 1

    # Save the final result
    # NOTE: This was previously {"final_resule": RESULT}
    scio.savemat(args.save_dir / 'final_result.mat', {"final_result": RESULT})
