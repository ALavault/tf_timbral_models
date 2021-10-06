import numpy as np
import tensorflow as tf
import glob
import os
from timbral_models import timbral_util
from timbral_models import timbral_roughness, tf_timbral_roughness
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
acm_score = np.array(timbral_roughness(fname, dev_output=False))

acm_score2 = np.array(timbral_roughness(fname2, dev_output=False))
tt = 128 * 128

fps = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
error = []
grad = True
for fname in fps:
    audio_samples, fs = timbral_util.file_read(
        fname, 0, phase_correction=False)
    audio_samples_t = tf.convert_to_tensor(
        [audio_samples[:tt], audio_samples[:tt]], dtype=tf.float32)
    audio_samples_t = tf.expand_dims(audio_samples_t, -1)
    acm_score = np.array(timbral_roughness(
        fname, dev_output=False, take_first=tt))
    if grad:
        with tf.GradientTape() as g:
            g.watch(audio_samples_t)
            tf_score = tf_timbral_roughness(
                audio_samples_t, fs=fs, dev_output=False)
        dy_dx = g.gradient(tf_score, audio_samples_t)
        assert dy_dx is not None, dy_dx
    tf_score = tf_timbral_roughness(
        audio_samples_t, fs=fs, dev_output=False)
    print("Score ::", acm_score, tf_score.numpy()[0])
    #error.append(100 * (acm_score - tf_score.numpy()) / acm_score)

error = np.array(error)
print("mean error :: {} %, std :: {}".format(np.mean(error), np.std(error)))
