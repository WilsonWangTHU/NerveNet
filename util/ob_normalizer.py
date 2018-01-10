# -----------------------------------------------------------------------------
#   @brief:
#       do a running mean and normalization on the observation
#       This is actually very important for the performances
#   @Updated:
#       Jun., 22rd, 2017, by Tingwu Wang
#       Aug., 8th, 2017, add the running mean by Tingwu Wang
#       Oct., 10th, 2017, add the transfer running mean
# -----------------------------------------------------------------------------


import numpy as np
import tensorflow as tf



class normalizer(object):
    '''
        @brief:
            The running mean of the input obs
    '''

    def __init__(self,
                 mean=0, variance=0, num_steps=0.,
                 filter_mean=True, clip_value=5.0):
        self.m1 = mean  # mean
        self.v = variance  # variance
        self.n = num_steps  # number of sample

        # options
        self.filter_mean = filter_mean
        if self.n < 1:
            self.std = 1
        else:
            self.std = (self.v + 1e-6) ** .5  # update the std

        self.clip_value = clip_value

    def raw_filter(self, o):
        '''
            @brief: using this filter won't change the statistics
        '''
        if self.filter_mean:
            o1 = (o - self.m1) / self.std
        else:
            o1 = o / self.std
        o1 = (o1 > self.clip_value) * self.clip_value \
            + (o1 < -self.clip_value) * (-self.clip_value) \
            + (o1 < self.clip_value) * (o1 > -self.clip_value) * o1
        return o1

    def filter(self, o, update=False):
        '''
            @in_batch:
                if it is set to true, then the first axis (dimention) will be
                reduce to mean
        '''
        if update:
            # update the parameters
            self.m1 = self.m1 * (self.n / (self.n + 1)) + o * 1 / (1 + self.n)
            self.v = self.v * (self.n / (self.n + 1)) + \
                (o - self.m1) ** 2 * 1 / (1 + self.n)
            self.std = (self.v + 1e-6) ** .5  # update the std
            self.n += 1

        return self.raw_filter(o)

    def set_parameters(self, mean, v, n):
        self.m1 = mean
        self.v = v
        self.n = n

        self.std = (self.v + 1e-6) ** .5  # update the std

    def get_parameters(self):
        return [self.m1, self.v, self.n]
