# -*- coding: utf-8 -*-
import numpy as np
import six


def get_standard_soft_pdf_type(pdf_type):
    ''' A function to define the softdata type in BME functions
    softpdf types can be specified by string or integer respectively

    softpdftype     PDF types                 associated parameters
    --------------------------------------------------------------------
    1               histogram                 (nl, limiB, probadens)
    2               linear                    (nl, limi, probadens)
    10              Gaussian or normal        (mean, var)
    
    More details: the way to define the flexible soft information follows 
    the definition of BMElib in Matlab as follows
    For a case of uniform distribution. For the histogram and linear-based 
    softdata can be expressed as below
    
    Histogram: softpdftype=1;nl=2;limi=[0,1];  probdens=[1]; 
    Linear: softpdftype=2;nl=2;limi=[0,1];  probdens=[1,1]
    
    '''
    if isinstance(pdf_type, six.string_types):
        if 'histogram'.startswith(pdf_type.lower()):
            return 1
        elif 'linear'.startswith(pdf_type.lower()):
            return 2
        elif 'gaussian'.startswith(pdf_type.lower()) or\
            'normal'.startswith(pdf_type.lower()):
            return 10
        else:
            raise ValueError('No supported pdftype found')
    else:
        if int(pdf_type) in [1]:
            return 1
        elif int(pdf_type) in [2] :
            return 2
        elif int(pdf_type) in [10] :
            return 10

def get_standard_order(order):
    '''
    A function to obtain the standard format of order parameter. The 
    order parameter determines the trend form in bme estimation.
    
    order  integer or    0 for constance mean and NaN for the zero mean   
           string        'zero mean' or 'constant mean' that explicitly describe 
                         the trend form
    
    Note: for S/T estimation, only zero and constant mean are supported for now 
          to avoid the overparametization of the trend modeling                                        
    
    '''
    ## For the string part, the string should be transformed captions into letters    
    ## before the comparison
    if isinstance(order, six.string_types): #base string
        if order.lower() == 'zero mean':
            return np.nan
        elif order.lower() == 'constant mean':
            return 0
        else:
            raise ValueError('No supported order (string) found')
    elif isinstance(order, int) or np.isnan(order): #int or np.nan
        if np.isnan(order):
            return order
        elif order in  [0, 0.0]:
            return 0
        else:
            raise ValueError('No supported order (number) found')
    else:
        print ('warning: "order" is not str or int, just return without modifing.')
        return order
