import os
import sys

import tensorflow as tf
 # tf.get_logger().setLevel('ERROR')

from data_loader.data_generator import DataGenerator
from models.signs_model import SignsModel
from trainers.signs_trainer import SignsTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
import numpy as np

np.random.seed(1)

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.compat.v1.Session()
    # create your data generator
    data = DataGenerator(config)
    
    # create an instance of the model you want
    model = SignsModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = SignsTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()

    train_acc = model.accuracy.eval({model.x: data.X_train,
                                     model.y: data.Y_train}, session=sess)
    test_acc = model.accuracy.eval({model.x: data.X_test,
                                    model.y: data.Y_test}, session=sess)
    print("Train Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)
    print("Done!")

if __name__ == '__main__':
    main()
