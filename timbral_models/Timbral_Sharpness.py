from __future__ import division
import numpy as np
from . import timbral_util
import tensorflow as tf
import tensorflow.keras.backend as K


def sharpness_Fastl(loudspec):
    """
      Calculates the sharpness based on FASTL (1991)
      Expression for weighting function obtained by fitting an
      equation to data given in 'Psychoacoustics: Facts and Models'
      using MATLAB basic fitting function
      Original Matlab code by Claire Churchill Sep 2004
      Transcoded by Andy Pearce 2018
    """
    n = len(loudspec)
    gz = np.ones(140)
    z = np.arange(141, n + 1)
    gzz = (
        0.00012 * (z / 10.0) ** 4
        - 0.0056 * (z / 10.0) ** 3
        + 0.1 * (z / 10.0) ** 2
        - 0.81 * (z / 10.0)
        + 3.5
    )
    gz = np.concatenate((gz, gzz))
    z = np.arange(0.1, n / 10.0 + 0.1, 0.1)

    sharp = 0.11 * np.sum(loudspec * gz * z * 0.1) / np.sum(loudspec * 0.1)
    return sharp


def tf_sharpness_Fastl(loudspec):
    """
      Calculates the sharpness based on FASTL (1991)
      Expression for weighting function obtained by fitting an
      equation to data given in 'Psychoacoustics: Facts and Models'
      using MATLAB basic fitting function
      Original Matlab code by Claire Churchill Sep 2004
      Transcoded by Andy Pearce 2018
    """
    n = loudspec.shape[-1]
    gz = tf.ones(140, dtype=loudspec.dtype)
    z = K.arange(141, n + 1, dtype=gz.dtype)
    gzz = (
        0.00012 * (z / 10.0) ** 4
        - 0.0056 * (z / 10.0) ** 3
        + 0.1 * (z / 10.0) ** 2
        - 0.81 * (z / 10.0)
        + 3.5
    )
    gz = tf.concat((gz, gzz), axis=-1)
    z = np.arange(0.1, n / 10.0 + 0.1, 0.1)

    sharp = (
        0.11 * K.sum(loudspec * gz * z * 0.1, axis=-1) /
        K.sum(loudspec * 0.1, axis=-1)
    )
    return sharp


def timbral_sharpness(
    fname,
    dev_output=False,
    phase_correction=False,
    clip_output=False,
    fs=0,
    take_first=None,
):
    """
     This is an implementation of the matlab sharpness function found at:
     https://www.salford.ac.uk/research/sirc/research-groups/acoustics/psychoacoustics/sound-quality-making-products-sound-better/accordion/sound-quality-testing/matlab-codes

     This function calculates the apparent Sharpness of an audio file.
     This version of timbral_sharpness contains self loudness normalising methods and can accept arrays as an input
     instead of a string filename.

     Version 0.4

     Originally coded by Claire Churchill Sep 2004
     Transcoded by Andy Pearce 2018

     Required parameter
      :param fname:                   string, audio filename to be analysed, including full file path and extension.

     Optional parameters
      :param dev_output:              bool, when False return the warmth, when True return all extracted features
      :param phase_correction:        bool, if the inter-channel phase should be estimated when performing a mono sum.
                                      Defaults to False.
      :param clip_output:             bool, bool, force the output to be between 0 and 100.  Defaults to False.

      :return                         Apparent sharpness of the audio file.


     Copyright 2018 Andy Pearce, Institute of Sound Recording, University of Surrey, UK.

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.

    """
    """
      Read input
    """
    audio_samples, fs = timbral_util.file_read(
        fname, fs, phase_correction=phase_correction
    )
    if take_first:
        audio_samples = audio_samples[:take_first]

    # window the audio file into 4096 sample sections
    windowed_audio = timbral_util.window_audio(
        audio_samples, window_length=4096)

    windowed_sharpness = []
    windowed_rms = []
    # print("true windowed audio ::", windowed_audio.shape)

    for i in range(windowed_audio.shape[0]):
        samples = windowed_audio[i, :]

        # calculate the rms and append to list
        windowed_rms.append(np.sqrt(np.mean(samples * samples)))

        # calculate the specific loudness
        N_entire, N_single = timbral_util.specific_loudness(
            samples, Pref=100.0, fs=fs, Mod=0
        )

        # calculate the sharpness if section contains audio
        if N_entire > 0:
            sharpness = tf.numpy_function(
                sharpness_Fastl, [N_single], tf.float64)
        else:
            sharpness = 0

        windowed_sharpness.append(sharpness)

    # convert lists to numpy arrays for fancy indexing
    windowed_rms = np.array(windowed_rms)
    windowed_sharpness = np.array(windowed_sharpness)
    # calculate the sharpness as the rms-weighted average of sharpness
    rms_sharpness = np.average(
        windowed_sharpness, weights=(windowed_rms * windowed_rms)
    )

    # take the logarithm to better much subjective ratings
    rms_sharpness = np.log10(rms_sharpness)

    if dev_output:
        return [rms_sharpness]
    else:

        all_metrics = np.ones(2)
        all_metrics[0] = rms_sharpness

        # coefficients from linear regression
        coefficients = [102.50508921364404, 34.432655185001735]

        # apply regression
        sharpness = np.sum(all_metrics * coefficients)

        if clip_output:
            sharpness = timbral_util.output_clip(sharpness)

        return sharpness


# works without tf.function
@tf.function
def tf_timbral_sharpness(
    audio_tensor, dev_output=False, phase_correction=False, clip_output=False, fs=0
):
    """
     Copyright 2021 Antoine Lavault, Apeira Technologies, France

     Licensed under the Apache License, Version 2.0 (the "License");
     you may not use this file except in compliance with the License.
     You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

     Unless required by applicable law or agreed to in writing, software
     distributed under the License is distributed on an "AS IS" BASIS,
     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
     See the License for the specific language governing permissions and
     limitations under the License.
    """
    assert fs, "fs should be provided to tf_timbral_sharpness"
    del phase_correction
    tf.debugging.assert_rank(
        audio_tensor, 3, message=None, name="tf_timbral_sharpness_rank_assert"
    )
    audio_samples_t, fs = audio_tensor[:, :, 0], fs
    sharpness = 0.0
    b = audio_samples_t.shape[0]
    rms_sharpness_array = []
    # windowed rms should be easy to implement in pure tensorflow
    # print("audio_samples_t", audio_samples_t)

    windowed_audio = timbral_util.tf_window_audio(
        audio_samples_t, window_length=4096)
    # print("windowed_audio", windowed_audio)
    windowed_rms = K.sqrt(K.mean(windowed_audio * windowed_audio, axis=-1))
    # Gradient here in windowed_rms and windowed_audio
    for i in range(b):
        windowed_sharpness = []
        # audio_samples = audio_samples_t[i]
        # window the audio file into 4096 sample sections
        windowed_audio_ = windowed_audio[i]
        # windowed_audio = tf.signal.frame(audio_samples, 4096, 4096, pad_end=True)
        # print(windowed_audio_.shape)
        for j in range(windowed_audio_.shape[0]):
            samples = windowed_audio_[j, :]
            # calculate the rms and append to list
            # calculate the specific loudness
            N_entire, N_single = tf.numpy_function(
                timbral_util.specific_loudness,
                [samples, 100.0, fs, 0],
                [tf.float64, tf.float64],
            )
            # calculate the sharpness if section contains audio
            N_single = tf.reshape(N_single, [240])
            if N_entire > 0:
                sharpness = tf.cast(tf_sharpness_Fastl(
                    N_single), audio_tensor.dtype)
            else:
                sharpness = 0.0
            windowed_sharpness.append(sharpness)

        # convert lists to numpy arrays for fancy indexing
        windowed_sharpness = tf.stack(windowed_sharpness)

        # calculate the sharpness as the rms-weighted average of sharpness
        # print(windowed_rms[i])
        # print(windowed_sharpness)
        rms_sharpness = timbral_util.tf_average(
            windowed_sharpness, windowed_rms[i] ** 2
        )

        # take the logarithm to better much subjective ratings
        rms_sharpness = timbral_util.log10(rms_sharpness)
        rms_sharpness_array.append(rms_sharpness)

    # rms_sharpness = tf.cast(np.array(rms_sharpness_array), audio_tensor.dtype)
    rms_sharpness = tf.stack(rms_sharpness_array)
    # Limit of gradient test
    if dev_output:
        return rms_sharpness
    else:
        """
        all_metrics = np.ones(2)
        all_metrics[0] = rms_sharpness

        # coefficients from linear regression
        coefficients = [102.50508921364404, 34.432655185001735]

        # apply regression
        sharpness = np.sum(all_metrics * coefficients) + sharpness
        """
        sharpness = 102.50508921364404 * rms_sharpness + 34.432655185001735
        if clip_output:
            sharpness = timbral_util.output_clip(sharpness)
    return sharpness
