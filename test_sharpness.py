import numpy as np
import tensorflow as tf
import glob
import os
from timbral_models import timbral_util
from timbral_models import timbral_sharpness, tf_timbral_sharpness
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
acm_score = np.array(timbral_sharpness(fname, dev_output=False))

acm_score2 = np.array(timbral_sharpness(fname2, dev_output=False))
tt = 128 * 128

fps = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
error = []
# Testing the batch compatibility
audio_samples_t = tf.convert_to_tensor(
    [audio_samples[:tt]], dtype=tf.float32)[0]
audio_samples_t2 = tf.convert_to_tensor(
    [audio_samples[:tt]], dtype=tf.float32)[0]
print(audio_samples_t)
audio_samples_t = tf.stack([audio_samples_t, audio_samples_t2])
print(audio_samples_t)

grad = True
dev_output = False
for fname in fps[:10]:
    audio_samples, fs = timbral_util.file_read(
        fname, 0, phase_correction=False)[:tt]
    audio_samples_t = tf.convert_to_tensor(
        [audio_samples[:tt], audio_samples[:tt]], dtype=tf.float32)
    audio_samples_t = tf.expand_dims(audio_samples_t, -1)
    acm_score = np.array(timbral_sharpness(
        fname, dev_output=dev_output, take_first=tt))
    if grad:
        with tf.GradientTape() as g:
            g.watch(audio_samples_t)
            tf_score = tf_timbral_sharpness(
                audio_samples_t, fs=fs, dev_output=dev_output
            )
        print(acm_score, tf_score)
        dy_dx = g.gradient(tf_score, audio_samples_t)
        assert dy_dx is not None, dy_dx
    else:
        tf_score = tf_timbral_sharpness(
            audio_samples_t, fs=fs, dev_output=dev_output)
        print(acm_score, tf_score)
    error.append(100 * (acm_score - tf_score) / acm_score)

error = np.array(error)
print("mean error :: {} %, std :: {}".format(np.mean(error), np.std(error)))
