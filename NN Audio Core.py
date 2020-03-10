# File to contain all of the settings and code that must be in sync across 
#   Data generation
#   NN training
#   Evaluation & testing


# TODO - add file name generation functions
# TODO - figure out import in collab


import librosa
import numpy as np
import cmath
import math
import os

# Constants and settings
DTYPE = 'float32'

WINDOW_SIZE = 55  # Has to be odd
TARGET_COL = WINDOW_SIZE//2

# stft values
N_FFT = 1024 # 512 recommended for speech, music typically 2048
FFT_BINS = 513

HOP_LENGTH = 128 # Required for good round trip quality

# Shared functions

def get_ft(wav):
    c = librosa.stft(wav, hop_length=HOP_LENGTH, n_fft=N_FFT)
    return c

def get_ft_from_file(file):
    filename = os.fsdecode(file)
    wav, rate = librosa.core.load(filename)
    return get_ft(wav)

def inv_ft(ft):
    return librosa.istft(ft, hop_length=HOP_LENGTH)

# This is an approximation - much better ways to compare voice quality exist, but this works fine
# This is for FT complex values - not samples and magnitudes!  
# So format is bins, samples and we only pay attention to relevant ones
def diff_ft(ft1, ft2):
    return np.average(abs(ft1[LOW_BIN:HIGH_BIN,:]-ft2[LOW_BIN:HIGH_BIN,:]))


################################################################################
# Pitch & sample encoding functions

# Low cut off at bin 3 ~65hz (C2) - seems like 125hz should be safe but keeping the low end
# High cut off bin 300 ~6500hz - 5000hz sounded OK, but keeping some just in case
# These are based on N_FFT 1024
LOW_BIN = 3 
HIGH_BIN = 280
SAMPLE_BINS = HIGH_BIN-LOW_BIN

freq = librosa.fft_frequencies(sr=22050, n_fft=N_FFT)
LOW_FREQ = freq[LOW_BIN]
HIGH_FREQ = freq[HIGH_BIN]
SAMPLE_OCTAVES = math.log(HIGH_FREQ/LOW_FREQ,2)
BINS_PER_OCTAVE = 20  # Tested empirically this sounds fine (15 was also OK so this should be safe)


# Number of bins in rescaled pitch representation of the ft
PITCH_BINS = math.floor(SAMPLE_OCTAVES * BINS_PER_OCTAVE)

# S = sample space, which is frequency  P = pitch space, which is rescaled so pitches are constant space appart
def S_ix(p_ix):
    return SAMPLE_OCTAVES * ((SAMPLE_BINS/SAMPLE_OCTAVES + 1)**(p_ix/PITCH_BINS) -1 )

def P_ix(s_ix):
    return PITCH_BINS * math.log((1+s_ix/SAMPLE_OCTAVES),2)/math.log((1 + SAMPLE_BINS/SAMPLE_OCTAVES),2)

# Project X onto returned array Y using a list of ascending ys of len(X) with maximum value of leng(Y), so that
# where multiple ys's map to a single index i in X, assign X[i] to Y[those ys's]
# where a single yx maps to multiple indices in X, assign X[these i's] of X[ys]
def squash_stretch(ys, X, Ylen):
    Y = np.zeros((Ylen))
    i = 0
    while i < (len(ys)-1):
        xh = i
        if (math.floor(ys[i+1]) > math.floor(ys[i]+1)):
            #print("Batch assigning X[i] to Y[ys[i]:ys[i+1]] ",i, ys[i], ys[i+1])
            Y[ys[i]:ys[i+1]] = X[i]
            i += 1
        else:
            while (math.floor(ys[i]) == math.floor(ys[xh])) and (xh < len(ys)-1):
                xh += 1
            if (xh > i+1):
                Y[ys[i]] = 0
                #print("Averaging X[i,xh] into Y[ys[i]] ", i, xh, ys[i])
                Y[ys[i]] = np.average(X[i:xh])
                i = xh
            else:
                #print("Straight map X[i] to Y[ys[i]] ", i, ys[i])
                Y[ys[i]] = X[i]
                i += 1
        #print (i, ys[i], ys[xh])
    return Y

# Takes samples and rescales to pitch (log of frequencies so space between same pitch is constant)
# Output is samples, then rescaled magnitudes in shape (sample_count, PITCH_BINS)
def pitch_scale(st):
    si = list(range(SAMPLE_BINS))
    pi = [math.floor(P_ix(x)) for x in si]
    pt = np.empty((st.shape[0], PITCH_BINS))
    for i in range(st.shape[0]):
        pt[i,:] = squash_stretch(pi, st[i,:], PITCH_BINS)
    return pt

def sample_scale(pt):
    pi = list(range(PITCH_BINS))
    si = [math.floor(S_ix(x)) for x in pi]
    st = np.empty((pt.shape[0],SAMPLE_BINS))
    for i in range(pt.shape[0]):
        st[i,:] = squash_stretch(si, pt[i,:], SAMPLE_BINS)
    return st

# Sample output is ft converted to magnitude and .T to get samples then ft
# Only returns 'interesting' samples - between LOW_ and HIGH_ bins, so output shape is (sample_count, SAMPLE_BINS)
def get_samples(file):
    wav, rate = librosa.core.load(file)
    samples = abs(get_ft(wav).T) # organized as bins, frames so we need to transpose them to frames, bins
    return samples[:,LOW_BIN:HIGH_BIN]

def rebuild_fft(samples, original_fft):
    fft = np.zeros((samples.shape[0], FFT_BINS))
    fft[:,LOW_BIN:HIGH_BIN] = samples
    vphase = np.vectorize(cmath.phase)
    o_phase = vphase(original_fft)
    mag = fft.T
    vrect = np.vectorize(cmath.rect)
    return vrect(mag, o_phase)
