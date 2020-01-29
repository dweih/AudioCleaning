# File to contain all of the settings and code that must be in sync across 
#   Data generation
#   NN training
#   Evaluation & testing


# TODO - add file name generation functions
# TODO - figure out import in collab


import librosa
import numpy as np
import cmath
import os
import keras.backend as K

# Constants and settings
DTYPE = 'float32'

WINDOW_SIZE = 55  # Has to be odd
TARGET_COL = WINDOW_SIZE//2

# stft values
#N_FFT = 1024 # 512 recommended for speech, music typically 2048
#FFT_BINS = 513

# cqt values
HOP_LENGTH = 128 # Required for good round trip quality

BINS_PER_OCTAVE = 12 * 5
FMIN = librosa.note_to_hz('C1') # Could probably cut further up, but doesn't work in librosa
OCTAVES = 8

FFT_BINS = OCTAVES * BINS_PER_OCTAVE # function of items below


# Shared functions

# Idea here is that we operate on magnitude, and will just use phase from the original noisy sample
# NOTE that this only seems to work well with stft, not cqt
def rebuild_fft(output, original_fft):
    vphase = np.vectorize(cmath.phase)
    o_phase = vphase(original_fft)
    mag = output.T
    vrect = np.vectorize(cmath.rect)
    return vrect(mag, o_phase)


def get_ft(wav):
    c = librosa.cqt(wav, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=OCTAVES*BINS_PER_OCTAVE, bins_per_octave=BINS_PER_OCTAVE)
    #c = librosa.stft(wav, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return c

def get_ft_from_file(file):
    filename = os.fsdecode(file)
    wav, rate = librosa.core.load(filename)
    return get_ft(wav)

def inv_ft(ft):
    return librosa.icqt(ft, hop_length=HOP_LENGTH, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
    #return librosa.istft(ft, hop_length=HOP_LENGTH)

# This is an approximation - much better ways to compare voice quality exist, but this works fine
def diff_ft(ft1, ft2):
    per_sample = np.sum(abs(ft1-ft2), axis=0)
    return np.average(per_sample)

# Returns magnitude and phase
def get_samples(file):
    wav, rate = librosa.core.load(file)
    ft = get_ft(wav)
    polar_vect = np.vectorize(cmath.polar)
    M, P = polar_vect(ft)
    # organized as bins, frames so we need to transpose first two axes to frames, bins
    samples = np.empty((M.shape[1],M.shape[0],2))
    samples[:,:,0] = M.T 
    samples[:,:,1] = P.T
    return samples 

def rebuild_cqt(output):
    rect_vect = np.vectorize(cmath.rect)
    cqt = rect_vect(output[:,:,0], output[:,:,1])
    return cqt.T
    

def magnitude(X, axis):
    X2 = K.square(X)
    M2 = K.sum(X2, axis=axis, keepdims=True)
    return K.sqrt(M2)

# Punish this for being away from target magnitude
# Magic number of 0.5 for relative importance of components vs. magnitude
def magnitude_loss(y_actual, y_predicted):
    ya_magnitude = magnitude(y_actual,2)
    yp_magnitude = magnitude(y_predicted,2)
    mag_loss = K.mean(K.square(ya_magnitude-yp_magnitude))
    component_loss = K.mean(K.square(y_actual-y_predicted))
    return mag_loss + 0.5 * component_loss

# Solution for problems reimporting with custom loss function from https://github.com/keras-team/keras/issues/5916
import keras.losses
keras.losses.magnitude_loss = magnitude_loss

# Cruft to evaluate deleting
#def filter(cqt):
#    cqt[0:BINS_PER_OCTAVE,:] = 0
#    return cqt

# build up as (bins, samples) then transpose to model view of (samples, bins)
#def targets_to_fft(targets):
#    fft = np.empty((targets.shape[0],targets.shape[1]//2), dtype='complex64')
#    for i in range(0, targets.shape[0]):
#        fft[i] = combine_target(targets[i])
#    return fft.T   # transpose
