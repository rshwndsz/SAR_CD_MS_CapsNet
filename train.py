import argparse
from pathlib import Path
import numpy as np
from keras import layers, models, optimizers, regularizers, constraints
from keras import backend as K
from keras import callbacks
import scipy.io as scio

from mscapsnet.prepare import readdata
from mscapsnet.losses import margin_loss
from mscapsnet.models import MSCapsNet


def train(args):
    # Create save_dir if not found
    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    # Read data from .mat files
    data, test_shuffle_number = readdata(args.image_file,
                                         args.label_file,
                                         train_nsamples=1000,
                                         validation_nsamples=1000,
                                         windowsize=args.windowsize,
                                         istraining=True)
    np.save(str(args.save_dir / 'test_shuffle_number.npy'),
            test_shuffle_number)

    # Save the index of training samples
    scio.savemat(str(args.save_dir / 'training_index.mat'),
                 {"index": test_shuffle_number})
    x_train, y_train = data[0], data[1]
    x_valid, y_valid = data[2], data[3]

    # Load model
    model = MSCapsNet(input_shape=[args.windowsize, args.windowsize, 3],
                      n_class=args.n_class,
                      num_routing=args.num_routing,
                      batch_size=args.batch_size)
    # Print summary
    # TODO Do this only on verbose mode
    model.summary()

    tb = callbacks.TensorBoard(log_dir=str(args.save_dir / 'tensorboard-logs'),
                               batch_size=args.batch_size)
    ckpt = callbacks.ModelCheckpoint(str(args.save_dir / 'weights-test.h5'),
                                     save_best_only=True,
                                     save_weights_only=True,
                                     verbose=1)
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss],
                  metrics={' ': 'accuracy'})
    model.fit(x_train,
              y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=[x_valid, y_valid],
              callbacks=[tb, ckpt],
              verbose=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--n_class',
                        help="Number of classes",
                        default=2,
                        type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--num_routing',
                        default=3,
                        help="Should be >= 1",
                        type=int)
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('--is_training', default=0, type=int)
    parser.add_argument('--lr',
                        help="Learning rate",
                        default=0.001,
                        type=float)
    parser.add_argument('--windowsize', help="Patch size", default=9, type=int)
    parser.add_argument('--image_file', default="./data/YellowRiverI.mat")
    parser.add_argument('--label_file', default="./data/YellowRiverI_gt.mat")
    args = parser.parse_args()

    train(args)
