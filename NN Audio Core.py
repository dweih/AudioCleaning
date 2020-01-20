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

# Constants and settings
DTYPE = 'float32'

WINDOW_SIZE = 55  # Has to be odd
TARGET_COL = WINDOW_SIZE//2

# stft values
N_FFT = 1024 # 512 recommended for speech, music typically 2048
FFT_BINS = 513

# cqt values
#FFT_BINS = 768 # function of items below
HOP_LENGTH = 256

BINS_PER_OCTAVE = 12 * 8
FMIN = librosa.note_to_hz('C1')
OCTAVES = 8


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
    #c = librosa.cqt(wav, hop_length=HOP_LENGTH, fmin=FMIN, n_bins=OCTAVES*BINS_PER_OCTAVE, bins_per_octave=BINS_PER_OCTAVE)
    c = librosa.stft(wav, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return c

def get_ft_from_file(file):
    filename = os.fsdecode(file)
    wav, rate = librosa.core.load(filename)
    return get_ft(wav)

def inv_ft(ft):
    #return librosa.icqt(ft, hop_length=HOP_LENGTH, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE)
    return librosa.istft(ft, hop_length=HOP_LENGTH)

# This is an approximation - much better ways to compare voice quality exist, but this works fine
def diff_ft(ft1, ft2):
    per_sample = np.sum(abs(ft1-ft2), axis=0)
    return np.average(per_sample)


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
