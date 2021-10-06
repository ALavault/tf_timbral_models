import numpy as np
from scipy import signal
import tensorflow as tf
import glob
import os
from timbral_models import filter_audio_highpass, tf_filter_audio_highpass, timbral_util
from timbral_models import timbral_brightness, tf_timbral_brightness_2
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

fname = (
    "/home/ubuntu/Documents/code/data/drummer_1_3_sd_001_hits_snare-drum_sticks_x6.wav"
)
fname2 = (
    "/home/ubuntu/Documents/code/data/drummer_3_0_tom_004_hits_low-tom-1_sticks_x5.wav"
)
data_dir = "/home/ubuntu/Documents/code/data/"
audio_samples, fs = timbral_util.file_read(fname, 0, phase_correction=False)
audio_samples2, fs = timbral_util.file_read(fname2, 0, phase_correction=False)

tt = 128 * 128
print("error :: {}\%".format(np.mean(25)))

