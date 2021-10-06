import glob
import logging
import os
import sys
import warnings

import numpy as np
import tensorflow as tf
from scipy import signal, stats

from timbral_models import (filter_audio_highpass, tf_filter_audio_highpass,
                            tf_timbral_sharpness, timbral_sharpness,
                            timbral_util)

try:
    import manage_gpus as gpl

    at_ircam = True
    print("manage_gpus detected. IRCAM computer here.")
except ImportError:
    gpl = None
    at_ircam = False
    max_num_threads = 6
    print("manage_gpus was not found. Assuming it is not an IRCAM computer.")
except Exception as inst:
    print("Unexpected error while importing manage_gpus. Exiting.")
    print(type(inst))  # the exception instance    exit(-1)
    raise inst

if not at_ircam:
    # tf.profiler.experimental.server.start(6009)
    pass

if gpl:
    try:
        gpu_id_locked = gpl.get_gpu_lock(gpu_device_id=-1, soft=True)
        print("GPU {} successfully locked !".format(gpu_id_locked))
    except gpl.NoGpuManager:
        print(
            "no gpu manager available - will use all available GPUs", file=sys.stderr
        )
    except gpl.NoGpuAvailable:
        print(
            "there is no GPU available for locking, continuing with CPU",
            file=sys.stderr,
        )
        comp_device = "/cpu:0"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")
data_dir = "/data/anasynth_nonbp/lavault/data_for_pickle_augmented_general_noc/data_hd_curated"

tt = 128 * 128

fps = glob.glob(os.path.join(data_dir, "**/*sd*.wav"), recursive=True)
error = []

timbral_util.print_blue(
    "Preparing the test. This may take a while. Please wait....")

grad = True
ll = len(fps)
for i, fname in enumerate(fps):
    audio_samples, fs = timbral_util.file_read(
        fname, 0, phase_correction=False)
    audio_samples_t = tf.convert_to_tensor(
        [audio_samples[:tt]], dtype=tf.float32)
    audio_samples_t = tf.expand_dims(audio_samples_t, -1)
    acm_score = np.array(timbral_sharpness(
        fname, dev_output=False, take_first=tt))
    tf_score_2 = tf_timbral_sharpness(
        audio_samples_t, fs=fs, dev_output=False,
    )
    error.append(tf_score_2.numpy())
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("{} / {} :: {:.2f}%".format(i+1, ll, ((i+1)/ll) * 100))
    sys.stdout.flush()


print("mean sharpness :: {} , std :: {}, min :: {}, max :: {}".format(
    np.mean(error), np.std(error), min(error), max(error)))
