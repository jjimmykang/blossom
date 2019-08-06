#
# This code is part of the Blossom project.
#
# Written by Jimmy Kang <jimmykang1016@gmail.com>, March 2019
#

import blossomai as bl
import tensorflow as tf
from datetime import datetime

log_file_location = '../logs/training_sessions.log'

feature_list = [
'masa_all', 'del_asa_all', 'sbr_all', 'rasa_all', 'del_rasa_all',
'blosum_a', 'blosum_c', 'blosum_d', 'blosum_e', 'blosum_f',
'blosum_g', 'blosum_h', 'blosum_i', 'blosum_k', 'blosum_l', 'blosum_m',
'blosum_n', 'blosum_p', 'blosum_q', 'blosum_r', 'blosum_s', 'blosum_t',
'blosum_v', 'blosum_w', 'blosum_y',
'hydrophobicity', 'hydrophilicity',
'polarity', 'polarizability', 'propensities', 'aasa', 'pssm_a',
'pssm_r', 'pssm_n', 'pssm_d', 'pssm_c', 'pssm_q', 'pssm_e', 'pssm_g',
'pssm_h', 'pssm_i', 'pssm_l', 'pssm_k', 'pssm_m', 'pssm_f', 'pssm_p',
'pssm_s', 'pssm_t', 'pssm_w', 'pssm_y', 'pssm_v', 'HSEBD', 'HSEAU', 'HSEAD',
'HSEBU', 'CN', 'RD', 'Rda'
]

class SessionResults:
    '''
    Results contained for BlossomSession
    '''
    training_success = 0
    validation_success = 0
    sensitivity_training = 0
    specificity_training = 0
    sensitivity_validation = 0
    specificity_validation = 0

    def display(self):
        print("Training Sucess Rate:", self.training_success)
        print("Validation Success Rate:", self.validation_success)
        print("Training Sensitivity:", self.sensitivity_training)
        print("Training Specificity:", self.specificity_training)
        print("Validation Sensitivity:", self.sensitivity_validation)
        print("Validation Specificity:", self.specificity_validation)



class BlossomSession:
    '''
    One specific instance of a blossom net trained.
    The variables represent all the possible aspects of the network to alter.
    '''
    used_features = []
    learning_rate = 0
    l1_strength = 0
    hidden_units = []
    reg_mult = 1
    results = SessionResults()
    trained = False
    improvement_list = []
    period_selected=10
    steps=50

    def __init__(self, features, rate, strength, units, multiplier, period, steps):
        '''
        features = list of indexes from the used_features list
        rate = any number between 0 and 1 to specify learning rate
        strength = any number between 0 and 1 to specify L1 regularization strength
        units = list of hidden units
        '''
        self.learning_rate = rate
        self.l1_strength = strength
        self.hidden_units = units
        self.reg_mult = multiplier
        self.period_selected = period
        self.steps = steps
        append_features = []
        for index in features:
            append_features.append(feature_list[index])
        self.used_features = append_features

    def parameters(self):
        print("used_features:", self.used_features)
        print("learning_rate:", self.learning_rate)
        print("l1_strength:", self.l1_strength)
        print("hidden_units:", self.hidden_units)
        print("periods:", self.period_selected)
        print("steps per period:", self.steps)

    def train(self):
        print("running blossom_run")
        raw_results = bl.blossom_run(used_features=self.used_features,
            user_learning_rate=self.learning_rate,
            l1_strength=self.l1_strength,
            hidden_unit_list=self.hidden_units,
            normal_mult=self.reg_mult,
            period_selected=self.period_selected)
        self.results.training_success = raw_results["training_success"]
        self.results.validation_success = raw_results["validation_success"]
        self.results.sensitivity_training = raw_results["sensitivity_training"]
        self.results.specificity_training = raw_results["specificity_training"]
        self.results.sensitivity_validation = raw_results["sensitivity_validation"]
        self.results.specificity_validation = raw_results["specificity_validation"]
        self.improvement_list = raw_results["improvement_list"]

        print("encoded results variables")
        self.trained = True

    def stats(self):
        if self.trained:
            self.results.display()
        else:
            print("Session is not trained yet.")

    def save(self):
        log_file = open(log_file_location, 'a')
        log_file.write("\n\n\n" + str(datetime.now()))
        log_file.write("\n")
        i=1
        for rate in self.improvement_list:
            log_file.write("period %02d : %0.2f\n" % (i, rate))
            i += 1
        log_file.write("\n" + "SETTINGS" + "\n")
        log_file.write("Used features: "+ str(self.used_features) + '\n')
        log_file.write("Learning rate: "+ str(self.learning_rate) + '\n')
        log_file.write("L1 Strength: "+ str(self.l1_strength) + '\n')
        log_file.write("Hidden Units: "+ str(self.hidden_units) + '\n')
        log_file.write("Periods: "+ str(self.period_selected) + '\n')
        log_file.write("Steps per period: "+ str(self.steps) + '\n')
        log_file.write("Regularization multiplier: "+ str(self.reg_mult))

        log_file.write("\n\n" + "STATISTICS" + "\n")
        log_file.write("Training Success Rate: "+ str(self.results.training_success) + '\n')
        log_file.write("Validation Success Rate: "+ str(self.results.validation_success) + '\n')
        log_file.write("Training Sensitivity: "+ str(self.results.sensitivity_training) + '\n')
        log_file.write("Training Specificity: "+ str(self.results.specificity_training) + '\n')
        log_file.write("Validation Sensitivity: "+ str(self.results.sensitivity_validation) + '\n')
        log_file.write("Validation Specificity: "+ str(self.results.specificity_validation))
        log_file.close()


primarySession = BlossomSession([0, 1, 2, 3, 4, 25, 26], #features
    0.12, #Learning rate
    0.05, #L1 Regularization Strength
    [7, 5, 3, 1], #Hidden unit list
    5, #normalization multiplier
    20, #periods
    500) #steps per period
primarySession.train()
primarySession.stats()
primarySession.save()
