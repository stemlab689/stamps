# -*- coding: utf-8 -*-
import numpy

class BMEoptions(object):
    def __init__(self):
        options = numpy.array([
            [0.],  # 1
            [0.0001],  # 2
            [10000.],  # 3 is the maximum number of evaluation of the integral
            [0.001],  # 4 is the relative error on the estimation of the integral
            [0.],  # 5 (test)
            [25.],  # 6
            [0.001],  # 7
            [2.],  # 8 number of moments calculated
                  # 1 for mean, 2 for mean and std dev, 3 for all three moments)
            [0.],  # 9 (test)
            [0.],  # 10 (test)
            [0.],  # 11 (test)
            [0.],  # 12 (test)
            [0.],  # 13 (test)
            [100.],  # 14
            [0.],  # 15 (test)
            [0.],  # 16 (test)
            [0.],  # 17 (test)
            [0.],  # 18 (test)
            [0.],  # 19 (test)
            [0.68],  # 20
            [0.],  # 21
            [0.],  # 22
            [0.],  # 23
            [0.],  # 24
            [0.],  # 25
            [0.],  # 26
            [0.],  # 27
            [0.],  # 28
            [0.],  # 29
            ])

        #make options pythonic
        self.options_dict = {}
        for idx, k in enumerate(options):
            self.options_dict[idx] = k
            self.options_dict[idx,0] = k[0]

        #we can add more options with key and value here
        #and old code still compatible
        self.options_dict['integration method'] = 'qmc'
        self.options_dict['qmc_showinfo'] = False
        self.options_dict['ck pdf debug'] = False
        self.options_dict['debug'] = False

    def __setitem__(self, key, value):
        if type(key) == tuple:
            self.options_dict[key] = value
            self.options_dict[key[0]] = [value]
        elif type(key) == str:
            self.options_dict[key] = value
        else:
            raise TypeError(
                'Not a valid key type: {t}'.format(t=type(key))
                )

    def __getitem__ (self, key):
        return self.options_dict[key]
