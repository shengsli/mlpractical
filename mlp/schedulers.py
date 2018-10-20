# -*- coding: utf-8 -*-
"""Training schedulers.

This module contains classes implementing schedulers which control the
evolution of learning rule hyperparameters (such as learning rate) over a
training run.
"""

import numpy as np


class ConstantLearningRateScheduler(object):
    """Example of scheduler interface which sets a constant learning rate."""

    def __init__(self, learning_rate):
        """Construct a new constant learning rate scheduler object.

        Args:
            learning_rate: Learning rate to use in learning rule.
        """
        self.learning_rate = learning_rate

    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        learning_rule.learning_rate = self.learning_rate

class CosineAnnealingWithWarmRestarts(object):
    """Cosine annealing scheduler, implemented as in https://arxiv.org/pdf/1608.03983.pdf"""

    def __init__(self, min_learning_rate, max_learning_rate, total_iters_per_period, max_learning_rate_discount_factor,
                 period_iteration_expansion_factor):
        """
        Instantiates a new cosine annealing with warm restarts learning rate scheduler
        :param min_learning_rate: The minimum learning rate the scheduler can assign
        :param max_learning_rate: The maximum learning rate the scheduler can assign
        :param total_epochs_per_period: The number of epochs in a period
        :param max_learning_rate_discount_factor: The rate of discount for the maximum learning rate after each restart i.e. how many times smaller the max learning rate will be after a restart compared to the previous one
        :param period_iteration_expansion_factor: The rate of expansion of the period epochs. e.g. if it's set to 1 then all periods have the same number of epochs, if it's larger than 1 then each subsequent period will have more epochs and vice versa.
        """
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.total_epochs_per_period = total_iters_per_period

        self.max_learning_rate_discount_factor = max_learning_rate_discount_factor
        self.period_iteration_expansion_factor = period_iteration_expansion_factor
        
    def update_learning_rule(self, learning_rule, epoch_number):
        """Update the hyperparameters of the learning rule.

        Run at the beginning of each epoch.

        Args:
            learning_rule: Learning rule object being used in training run,
                any scheduled hyperparameters to be altered should be
                attributes of this object.
            epoch_number: Integer index of training epoch about to be run.
        """
        #number of epochs in finished periods
        period_number = epoch_number//self.total_epochs_per_period
        discount = self.max_learning_rate_discount_factor**period_number
        cur_learning_rate = self.max_learning_rate * discount
        #how many epochs have been performed since the last restart
        T_cur = epoch_number % self.total_epochs_per_period
        #restart after T_i epochs are performed
        T_i = self.total_epochs_per_period * self.period_iteration_expansion_factor ** (period_number+1)
        learning_rule.learning_rate = self.min_learning_rate + 0.5*(cur_learning_rate-self.min_learning_rate)*(1+np.cos(np.pi*T_cur/T_i))
        return learning_rule.learning_rate
        

#         temp = (epoch_number//self.total_epochs_per_period)+1
#         period_number = np.log(temp**self.period_iteration_expansion_factor)/np.log(temp)
        
#         discount = self.max_learning_rate_discount_factor**period_number
#         self.max_learning_rate -= discount
#         #how many epochs have been performed since the last restart
#         T_cur = epoch_number - self.num_of_epochs_in_first_n_periods(period_number)
#         #restart after T_i epochs are performed
#         T_i = self.num_of_epochs_in_n_th_period(period_number)

#         learning_rate = self.min_learning_rate + 0.5*(self.max_learning_rate-self.min_learning_rate)*(1+np.cos(np.pi*T_cur/T_i))

#     def num_of_epochs_in_first_n_periods(self,n):
#         return (self.period_iteration_expansion_factor**n-1)*self.total_epochs_per_period
#     def num_of_epochs_in_n_th_period(self,n):
#         return self.total_epochs_per_period*self.period_iteration_expansion_factor**(n-1)