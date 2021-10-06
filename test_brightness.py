from timbral_models import timbral_brightness, tf_timbral_brightness_2
from timbral_models import filter_audio_highpass, tf_filter_audio_highpass, timbral_util
import tensorflow as tf
import warnings
import glob
import numpy as np
from scipy import signal, stats
import os
import logging
import librosa

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")

data_dir = "/home/ubuntu/Documents/code/data/"

tt = 128 * 128

fps = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
error = []
error2 = []

timbral_util.print_blue(
    "Preparing the test. This may take a while. Please wait....")

grad = True
value = []
for fname in fps:
    audio_samples, fs = timbral_util.file_read(
        fname, 0, phase_correction=False)
    if "sd" in fname:
        ds = librosa.resample(audio_samples, fs, 16000)
        acm_score_ds = np.array(timbral_brightness(
            None, fs=16000, dev_output=False, take_first=tt, in_audio=ds))
    audio_samples_t = tf.convert_to_tensor(
        [audio_samples[:tt]], dtype=tf.float32)
    audio_samples_t = tf.expand_dims(audio_samples_t, -1)
    acm_score = np.array(timbral_brightness(
        fname, dev_output=False, take_first=tt))

    if grad:
        with tf.GradientTape() as g:
            g.watch(audio_samples_t)
            tf_score = tf_timbral_brightness_2(
                audio_samples_t, fs=fs, dev_output=False)
        dy_dx = g.gradient(tf_score, audio_samples_t)
        assert (
            dy_dx is not None
        ), "Got a None in the gradients. Please check the function."
    else:
        tf_score = tf_timbral_brightness_2(
            audio_samples_t, fs=fs, dev_output=False)
    tf_score_2 = tf_timbral_brightness_2(
        audio_samples_t, fs=fs, dev_output=False, compensated=True
    )
    error.append(100 * (acm_score - tf_score.numpy()) / acm_score)
    if "sd" in fname:
        error2.append(100 * (acm_score - acm_score_ds) / acm_score)
        value.append(acm_score_ds)
    print("mean error :: {}, {}, {}".format(
        acm_score, acm_score_ds, tf_score_2.numpy()))


error = np.array(error)
error2 = np.array(error2)
value = np.array(value)
print("mean error :: {} %, std :: {}".format(np.mean(error), np.std(error)))

print(
    "mean error compensated :: {},{} %, std {},{} :: min {}, max {}".format(
        np.mean(error2), np.mean(value), np.std(error2), np.std(value), np.min(value), np.max(value))

)
