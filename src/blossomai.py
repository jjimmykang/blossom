#
# This code is part of the Blossom project.
#
# Written by Jimmy Kang <jimmykang1016@gmail.com>, March 2019
#


from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

import copy
import random
from datetime import datetime

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

master_training_location = '../review_data/master_training.csv'
master_testing_location = '../review_data/master_testing.csv'


def read_file():
    master_training_dataframe = pd.read_csv(master_training_location, sep=',', index_col=0)
    master_testing_dataframe = pd.read_csv(master_testing_location, sep=',', index_col=0)

    return (master_training_dataframe, master_testing_dataframe)



#Normalization Code

def normalize_dataframe(input_dataframe, multiplier=1):
    '''Normalizes any dataframe via mean/std in the column
    '''


    columns = list(input_dataframe)
    return_dataframe = input_dataframe.copy()
    stats_dict = {}

    def zscore(mean, std, value):
        return (value-mean) / std

    def stats_column(column):
        '''Returns a dictionary of mean/std
        '''
        mean = input_dataframe[column].mean()
        std = input_dataframe[column].std()
        return {'mean': mean, 'std': std}

    for i in columns:
        stats_dict.update({i : stats_column(i)})

    for column in columns:
        if column != 'hotspot':
            index = 0
            for item in input_dataframe[column]:
                item_zscore = zscore(stats_dict[column]['mean'], stats_dict[column]['std'], item)
                return_dataframe.loc[index, column] = item_zscore * multiplier
                index += 1

    return return_dataframe



def preprocess_features(master_dataframe, used_features):
    """Prepares input features from the master training dataset.

    Args:
      master_dataframe: a pd dataframe that should contain the master training data.
    Returns:
      A DataFrame that contains the features to be used for the model, including
      synthetic features.
    """

    #Select features to be used here
    selected_features = master_dataframe[used_features]

    #copy to processed feature list
    processed_features = selected_features.copy()

    #NOTE: If synthetic features are desired, you can add them HERE

    return processed_features

def preprocess_targets(master_dataframe):
    """Prepares target features (i.e., labels) from master training dataset.

    Args:
      master_dataframe: a pd dataframe that should contain the ASA training data.
    Returns:
      A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    output_targets["hotspot"] = master_dataframe["hotspot"]
    return output_targets

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
      input_features: The names of the numerical input features to use.
    Returns:
      A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def input_fn(features, targets, batch_size = 1, shuffle=False, num_epochs = None):
    """Trains a neural network model.
    NOTE: This code is from Google's tensorflow tutorial.
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_classification_model(
    my_optimizer,
    steps,
    periods,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    activation_function):
    """Trains a neural network classification model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A dataframe containing the features to predict the hotspot value of each instance.
      training_targets: A dataframe containing one column, and the hotspot values to target.
      validation_examples: A `DataFrame` containing one or more columns to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column to use as target for validation.

    Returns:
      A tuple `(estimator, training_losses, validation_losses)`:
        estimator: the trained `DNNClassifier` object.
        training_losses: a `list` containing the training loss values taken during training.
        validation_losses: a `list` containing the validation loss values taken during training.
    """


    steps_per_period = steps


    # Create input functions.
    print("creating input functions...")
    training_input_fn = lambda: input_fn(training_examples,
                                            training_targets["hotspot"],
                                            batch_size=batch_size)
    print("TRAINING TARGETS")
    print(training_targets)
    predict_training_input_fn = lambda: input_fn(training_examples,
                                                    training_targets["hotspot"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: input_fn(validation_examples,
                                                      validation_targets["hotspot"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Create a DNNClassifier object
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    print("creating DNNClassifier object...")
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=construct_feature_columns(training_examples),
        model_dir = "../models/" + str(datetime.now()), 
        hidden_units = hidden_units,
        optimizer = my_optimizer,
        activation_fn=activation_function,
        #model_dir='models/blossom'
    )


    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("Success Rate (on training data):")

    training_success = []
    validation_success = []

    training_correct = 0
    training_total = 0
    validation_correct = 0
    validation_total = 0

    true_positive_training = 0
    false_positive_training = 0

    true_negative_training = 0
    false_negative_training = 0

    true_positive_validation = 0
    false_positive_validation = 0

    true_negative_validation = 0
    false_negative_validation = 0

    improvement_list = []

    run = True
    while run:
        for period in range (0, periods):
            # Train the model, starting from the prior state.
            dnn_classifier.train(
                input_fn=training_input_fn,
                steps=steps_per_period
            )

            # Take a break and compute predictions.
            training_predictions = dnn_classifier.predict(input_fn=predict_training_input_fn)
            training_predictions = [int(item['classes']) for item in list(training_predictions)]

            validation_predictions = dnn_classifier.predict(input_fn=predict_validation_input_fn)
            validation_predictions = [int(item['classes']) for item in list(validation_predictions)]

            # Compute training and validation success rate
            for i in range(0, len(training_predictions)):
                if training_predictions[i] == training_targets['hotspot'][i]:
                    training_correct += 1
                    if training_predictions[i] == 1:
                        true_positive_training += 1
                    else:
                        true_negative_training += 1
                else:
                    if training_predictions[i] == 1:
                        false_positive_training += 1
                    else:
                        false_negative_training += 1
                training_total += 1
            training_success_rate = training_correct / training_total


            for i in range(0, len(validation_predictions)):
                if validation_predictions[i] == validation_targets['hotspot'][i]:
                    validation_correct += 1
                    if validation_predictions[i] == 1:
                        true_positive_validation += 1
                    else:
                        true_negative_validation += 1
                else:
                    if validation_predictions[i] == 1:
                        false_positive_validation += 1
                    else:
                        false_negative_validation += 1
                validation_total += 1
            validation_success_rate = validation_correct / validation_total

            # Occasionally print the current loss.
            print("  period %02d : %0.2f" % (period, training_success_rate))
            # Add the loss metrics from this period to our list.
            training_success.append(training_success_rate)
            validation_success.append(validation_success_rate)
            improvement_list.append(training_success_rate)

        ask = True
        while ask:
            indicate = str(input("Continue?(y/n):"))
            if indicate == 'y' or indicate == 'Y':
                run = True
                ask = False
            elif indicate == 'n' or indicate == 'N':
                run = False
                ask = False

    '''
    print("Model training finished.")
    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    '''

    sensitivity_training = true_positive_training / (true_positive_training + false_negative_training)
    specificity_training = true_negative_training / (true_negative_training + false_positive_training)

    sensitivity_validation = true_positive_validation / (true_positive_validation + false_negative_validation)
    specificity_validation = true_negative_validation / (true_negative_validation + false_positive_validation)

    final_training_success = training_success[-1]

    final_validation_success = validation_success[-1]

    #print("Final success rate(on training data):   %0.2f" % training_success_rate)
    #print("Final success rate (on validation data): %0.2f" % validation_success_rate)
    #print("Training Sensitivity:", sensitivity_training)
    #print("Training Specificity:", specificity_training)
    #print("Validation Sensitivity:", sensitivity_validation)
    #print("Validation Specificity:", specificity_validation)
    #print("Model training finished")



    return {"dnn_classifier": dnn_classifier, "training_success": final_training_success, "validation_success": final_validation_success,
        "sensitivity_training": sensitivity_training, "specificity_training": specificity_training,
        "sensitivity_validation": sensitivity_validation, "specificity_validation": specificity_validation, "improvement_list": improvement_list}


continue_bool = False
input_bool = True

'''
while input_bool:
    user_cont = input("Train model?(y/n):")
    if user_cont == 'y' or user_cont == 'Y':
        continue_bool = True
        input_bool = False
    elif user_cont == 'n' or user_cont == 'N':
        continue_bool = False
        input_bool = False
'''

def test_function():
    print("this works")


def blossom_run(used_features, user_learning_rate=0.07, l1_strength=0.07, hidden_unit_list=[7, 5, 3, 1], normal_mult=1, period_selected=10, steps = 50, input_activation=tf.nn.relu):

    master_training_dataframe = normalize_dataframe(read_file()[0], normal_mult)
    master_testing_dataframe = normalize_dataframe(read_file()[1], normal_mult)

    print(master_training_dataframe.shape)
    print(master_testing_dataframe.shape)

    print(master_training_dataframe.head())
    training_examples = preprocess_features(master_training_dataframe, used_features)
    training_targets = preprocess_targets(master_training_dataframe)

    validation_examples = preprocess_features(master_testing_dataframe, used_features)
    validation_targets = preprocess_targets(master_testing_dataframe)

    user_learning_rate = 0.07
    l1_strength = 0.07
    hidden_unit_list = [7, 5, 3, 1]

    _ = train_nn_classification_model(
        my_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=user_learning_rate, l1_regularization_strength = l1_strength),
        steps=steps,
        periods=period_selected,
        batch_size=70,
        hidden_units=hidden_unit_list,
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets,
        activation_function=input_activation)

    return _
