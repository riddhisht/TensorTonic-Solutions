import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):

    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None,:]

    angle_rates = pos / (base ** (2 * (i//2)/d_model))

    pe = np.zeros((seq_len, d_model))
    
    pe[:,0::2] = np.sin(angle_rates[:,0::2])
    pe[:, 1::2] = np.cos(angle_rates[:,1::2])

    return pe
    