from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from scipy.signal import spectrogram
from sklearn import linear_model

from . import timbral_util


def warm_region_cal(audio_samples, fs):
    """
      Function for calculating various warmth parameters.

    :param audio_samples:   numpy.array, an array of the audio samples, reques only one dimension.
    :param fs:              int, the sample ratr of the audio file.

    :return:                four outputs: mean warmth region, weighted-average warmth region, mean high frequency level,
                            weighted-average high frequency level.
    """
    # window the audio
    windowed_samples = timbral_util.window_audio(audio_samples)

    # need to define a function for the roughness stimuli, emphasising the 20 - 40 region (of the bark scale)
    min_bark_band = 10
    max_bark_band = 40
    mean_bark_band = (min_bark_band + max_bark_band) / 2.0
    array = np.arange(min_bark_band, max_bark_band)
    x = timbral_util.normal_dist(array, theta=0.01, mean=mean_bark_band)
    x -= np.min(x)
    x /= np.max(x)

    wr_array = np.zeros(240)
    wr_array[min_bark_band:max_bark_band] = x

    # need to define a second array emphasising the 20 - 40 region (of the bark scale)
    min_bark_band = 80
    max_bark_band = 240
    mean_bark_band = (min_bark_band + max_bark_band) / 2.0
    array = np.arange(min_bark_band, max_bark_band)
    x = timbral_util.normal_dist(array, theta=0.01, mean=mean_bark_band)
    x -= np.min(x)
    x /= np.max(x)

    hf_array = np.zeros(240)
    hf_array[min_bark_band:max_bark_band] = x

    windowed_loud_spec = []
    windowed_rms = []

    wr_vals = []
    hf_vals = []

    for i in range(windowed_samples.shape[0]):
        samples = windowed_samples[i, :]
        _, N_single = timbral_util.specific_loudness(
            samples, Pref=100.0, fs=fs, Mod=0)

        # append the loudness spec
        windowed_loud_spec.append(N_single)
        windowed_rms.append(np.sqrt(np.mean(samples * samples)))

        wr_vals.append(np.sum(wr_array * N_single))
        hf_vals.append(np.sum(hf_array * N_single))

    mean_wr = np.mean(wr_vals)
    mean_hf = np.mean(hf_vals)
    weighted_wr = np.average(wr_vals, weights=windowed_rms)
    weighted_hf = np.average(hf_vals, weights=windowed_rms)

    return mean_wr, weighted_wr, mean_hf, weighted_hf


def tf_warm_region_cal(audio_samples, fs, dtype=None):

    windowed_samples = timbral_util.tf_window_audio(
        audio_samples, 4096)
    # need to define a function for the roughness stimuli, emphasising the 20 - 40 region (of the bark scale)
    min_bark_band = 10
    max_bark_band = 40
    mean_bark_band = (min_bark_band + max_bark_band) / 2.0
    array = np.arange(min_bark_band, max_bark_band)
    x = timbral_util.normal_dist(array, theta=0.01, mean=mean_bark_band)
    x -= np.min(x)
    x /= np.max(x)

    wr_array = np.zeros(240)
    wr_array[min_bark_band:max_bark_band] = x

    # need to define a second array emphasising the 20 - 40 region (of the bark scale)
    min_bark_band = 80
    max_bark_band = 240
    mean_bark_band = (min_bark_band + max_bark_band) / 2.0
    array = np.arange(min_bark_band, max_bark_band)
    x = timbral_util.normal_dist(array, theta=0.01, mean=mean_bark_band)
    x -= np.min(x)
    x /= np.max(x)

    hf_array = np.zeros(240)
    hf_array[min_bark_band:max_bark_band] = x

    windowed_loud_spec = []
    windowed_rms = []

    wr_vals = []
    hf_vals = []
    windowed_rms = K.sqrt(K.mean(windowed_samples * windowed_samples, axis=-1))

    for i in range(windowed_samples.shape[0]):
        samples = windowed_samples[i, :]
        _, N_single = tf.numpy_function(
            timbral_util.specific_loudness,
            [samples, 100.0, fs, 0],
            [tf.float64, tf.float64],
            name="specific_loudness"
        )

        # append the loudness spec
        windowed_loud_spec.append(N_single)

        wr_vals.append(K.sum(wr_array * N_single))
        hf_vals.append(K.sum(hf_array * N_single))
    wr_vals = tf.cast(tf.stack(wr_vals), audio_samples.dtype)
    hf_vals = tf.cast(tf.stack(hf_vals), audio_samples.dtype)
    mean_wr = K.mean(wr_vals)
    mean_hf = K.mean(hf_vals)
    weighted_wr = timbral_util.tf_average(wr_vals, windowed_rms)
    weighted_hf = timbral_util.tf_average(hf_vals, windowed_rms)

    return mean_wr, weighted_wr, mean_hf, weighted_hf


def timbral_warmth(
    fname,
    dev_output=False,
    phase_correction=False,
    clip_output=False,
    max_FFT_frame_size=8192,
    max_WR=12000,
    fs=0,
    take_first=None

):
    """
     This function estimates the perceptual Warmth of an audio file.

     This model of timbral_warmth contains self loudness normalising methods and can accept arrays as an input
     instead of a string filename.

     Version 0.4

     Required parameter
    :param fname:                   string, Audio filename to be analysed, including full file path and extension.

    Optional parameters
    :param dev_output:              bool, when False return the warmth, when True return all extracted features in a
                                    list.
    :param phase_correction:        bool, if the inter-channel phase should be estimated when performing a mono sum.
                                    Defaults to False.
    :param max_FFT_frame_size:      int, Frame size for calculating spectrogram, default to 8192.
    :param max_WR:                  float, maximun allowable warmth region frequency, defaults to 12000.

    :return:                        Estimated warmth of audio file.

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
    # get the weighted high frequency content
    mean_wr, _, _, weighted_hf = warm_region_cal(audio_samples, fs)

    # calculate the onsets
    envelope = timbral_util.sample_and_hold_envelope_calculation(
        audio_samples, fs, decay_time=0.1
    )
    # envelope_time = np.arange(len(envelope)) / float(fs)

    # calculate the onsets
    nperseg = 4096
    original_onsets = timbral_util.calculate_onsets(
        audio_samples, envelope, fs, nperseg=nperseg
    )
    # If onsets don't exist, set it to time zero
    if not original_onsets:
        original_onsets = [0]
    # set to start of file in the case where there is only one onset
    if len(original_onsets) == 1:
        original_onsets = [0]
    original_onsets = [0]

    """
      Initialise lists for storing features
    """
    # set defaults for holding
    all_rms = []
    all_ratio = []
    all_SC = []
    # all_WR_Ratio = []
    all_decay_score = []

    # calculate metrics for each onset
    for idx, onset in enumerate(original_onsets):
        if onset == original_onsets[-1]:
            # this is the last onset
            segment = audio_samples[onset:]
        else:
            segment = audio_samples[onset: original_onsets[idx + 1]]

        segment_rms = np.sqrt(np.mean(segment * segment))
        all_rms.append(segment_rms)

        # get FFT of signal
        segment_length = len(segment)
        if segment_length < max_FFT_frame_size:
            freq, _, spec = spectrogram(
                segment, fs, nperseg=segment_length, nfft=max_FFT_frame_size
            )
        else:
            freq, _, spec = spectrogram(
                segment, fs, nperseg=max_FFT_frame_size, nfft=max_FFT_frame_size
            )

            # flatten the audio to 1 dimension.  Catches some strange errors that cause crashes
            if spec.shape[1] > 1:
                spec = np.sum(spec, axis=1)
                spec = spec.flatten()

        # normalise for this onset
        spec = np.array(list(spec)).flatten()
        # this_shape = spec.shape
        spec /= max(abs(spec))

        """
          Estimate of fundamental frequency
        """
        # peak picking algorithm
        peak_idx, _, peak_x = timbral_util.detect_peaks(spec, freq=freq, fs=fs)
        # find lowest peak
        fundamental = np.min(peak_x)
        fundamental_idx = np.min(peak_idx)

        """
         Warmth region calculation
        """
        # estimate the Warmth region
        WR_upper_f_limit = fundamental * 3.5
        if WR_upper_f_limit > max_WR:
            WR_upper_f_limit = 12000
        tpower = np.sum(spec)
        WR_upper_f_limit_idx = int(np.where(freq > WR_upper_f_limit)[0][0])

        if fundamental < 260:
            # find frequency bin closest to 260Hz
            top_level_idx = int(np.where(freq > 260)[0][0])
            # sum energy up to this bin
            low_energy = np.sum(spec[fundamental_idx:top_level_idx])
            # sum all energy
            tpower = np.sum(spec)
            # take ratio
            ratio = low_energy / float(tpower)
        else:
            # make exception where fundamental is greater than
            ratio = 0

        all_ratio.append(ratio)

        """
         Spectral centroid of the segment
        """
        # spectral centroid
        # top = np.sum(freq * spec)
        # bottom = float(np.sum(spec))
        SC = np.sum(freq * spec) / float(np.sum(spec))
        all_SC.append(SC)

        """
         HF decay
         - linear regression of the values above the warmth region
        """
        above_WR_spec = np.log10(spec[WR_upper_f_limit_idx:])
        above_WR_freq = np.log10(freq[WR_upper_f_limit_idx:])
        np.ones_like(above_WR_freq)
        metrics = np.array([above_WR_freq, np.ones_like(above_WR_freq)])

        # create a linear regression model
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(metrics.transpose(), above_WR_spec)
        decay_score = model.score(metrics.transpose(), above_WR_spec)
        all_decay_score.append(decay_score)

    """
     get mean values
    """
    mean_SC = np.log10(np.mean(all_SC))
    mean_decay_score = np.mean(all_decay_score)
    weighted_mean_ratio = np.average(all_ratio, weights=all_rms)

    if dev_output:
        return mean_SC, weighted_hf, mean_wr, mean_decay_score, weighted_mean_ratio
    else:

        """
         Apply regression model
        """
        all_metrics = np.ones(6)
        all_metrics[0] = mean_SC
        all_metrics[1] = weighted_hf
        all_metrics[2] = mean_wr
        all_metrics[3] = mean_decay_score
        all_metrics[4] = weighted_mean_ratio

        coefficients = np.array(
            [
                -4.464258317026696,
                -0.08819320850778556,
                0.29156539973575546,
                17.274733561081554,
                8.403340066029507,
                45.21212125085579,
            ]
        )

        warmth = np.sum(all_metrics * coefficients)

        # clip output between 0 and 100
        if clip_output:
            warmth = timbral_util.output_clip(warmth)

        return warmth


@tf.function
def tf_timbral_warmth(
    audio_tensor,
    dev_output=False,
    phase_correction=False,
    clip_output=False,
    max_FFT_frame_size=8192,
    max_WR=12000,
    fs=0,
):
    """
     This function estimates the perceptual Warmth of an audio file.

     This model of timbral_warmth contains self loudness normalising methods and can accept arrays as an input
     instead of a string filename.

     Version 0.4

     Required parameter
    :param fname:                   string, Audio filename to be analysed, including full file path and extension.

    Optional parameters
    :param dev_output:              bool, when False return the warmth, when True return all extracted features in a
                                    list.
    :param phase_correction:        bool, if the inter-channel phase should be estimated when performing a mono sum.
                                    Defaults to False.
    :param max_FFT_frame_size:      int, Frame size for calculating spectrogram, default to 8192.
    :param max_WR:                  float, maximun allowable warmth region frequency, defaults to 12000.

    :return:                        Estimated warmth of audio file.

    Copyright 2018 Andy Pearce, Institute of Sound Recording, University of Surrey, UK.
    Copyright 2020 Antoine Lavault, Apeira Technogloies, FR

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

      Read input
    """
    del phase_correction
    assert fs, "fs should be provided to tf_timbral_roughness"
    audio_samples, fs = audio_tensor, fs
    audio_samples = audio_samples[:, :, 0]
    fs = float(fs)
    all_decay_score_array = []
    mean_SC_array = []
    weighted_hf_array = []
    mean_wr_array = []
    mean_decay_score_array = []
    weighted_mean_ratio_array = []

    mean_decay_score = []
    weighted_mean_ratio = []
    b = audio_samples.shape[0]
    # same option as scipy spectrogram
    hop_size = int(max_FFT_frame_size / 8)
    # get the weighted high frequency content
    for k in range(b):
        all_decay_score = []
        samples = audio_samples[k]
        mean_wr, _, _, weighted_hf = tf_warm_region_cal(samples, fs)

        # calculate the onsets
        """
        envelope = timbral_util.sample_and_hold_envelope_calculation(
            audio_samples, fs, decay_time=0.1
        )
        """
        # envelope_time = np.arange(len(envelope)) / float(fs)

        # calculate the onsets
        """
        original_onsets = timbral_util.tf_calculate_onsets(
            audio_samples, envelope, fs, nperseg=nperseg
        )
        # If onsets don't exist, set it to time zero
        if not original_onsets:
            original_onsets = [0]
        # set to start of file in the case where there is only one onset
        if len(original_onsets) == 1:
            original_onsets = [0]
        """
        # tf_timbral onsets is 0 by default == using with gan4drum
        original_onsets = [0]
        """
        Initialise lists for storing features
        """
        # set defaults for holding
        all_rms = []
        all_ratio = []
        all_SC = []
        # all_WR_Ratio = []

        # calculate metrics for each onset
        # Hypothesis : used with drum samples : onset index = 0
        for idx, onset in enumerate(original_onsets):
            if onset == original_onsets[-1]:
                # this is the last onset
                segment = samples[onset:]
            else:
                segment = samples[onset: original_onsets[idx + 1]]

            segment_rms = K.sqrt(K.mean(segment * segment))
            all_rms.append(segment_rms)

            # get FFT of signal
            segment_length = len(segment)
            segment = segment[None, :]
            if segment_length < max_FFT_frame_size:
                freq, _, spec = timbral_util.compat_spectrogram(
                    segment, fs, nperseg=segment_length, nfft=max_FFT_frame_size, noverlap=hop_size
                )
            else:
                freq, _, spec = timbral_util.compat_spectrogram(
                    segment, fs, nperseg=max_FFT_frame_size, nfft=max_FFT_frame_size, noverlap=hop_size
                )

                # flatten the audio to 1 dimension.  Catches some strange errors that cause crashes
                if spec.shape[1] > 1:
                    spec = K.sum(spec, axis=1)
                    # spec = spec.flatten()
                    spec = tf.squeeze(spec)
            print("freq", freq)

            # normalise for this onset
            print(spec.dtype)
            spec = tf.squeeze(spec)
            # this_shape = spec.shape
            spec /= K.max(K.abs(spec))

            """
            Estimate of fundamental frequency
            """

            # peak picking algorithm
            #
            peak_idx, _, peak_x = tf.numpy_function(timbral_util.detect_peaks, [
                spec, freq, 0.2, spec, fs], [tf.int64, tf.float64, tf.float64],  name="detect_peaks")
            # find lowest peak
            fundamental = tf.cast(K.min(peak_x), audio_samples.dtype)
            fundamental_idx = tf.cast(K.min(peak_idx), tf.int32)

            """
            Warmth region calculation
            """
            # estimate the Warmth region
            WR_upper_f_limit = fundamental * 3.5
            if WR_upper_f_limit > max_WR:
                WR_upper_f_limit = float(12000)
            tpower = K.sum(spec)
            WR_upper_f_limit_idx = int(tf.where(freq > WR_upper_f_limit)[0][0])

            if fundamental < 260:
                # find frequency bin closest to 260Hz
                top_level_idx = int(tf.where(freq > 260)[0][0])
                # sum energy up to this bin
                low_energy = K.sum(spec[fundamental_idx:top_level_idx])
                # sum all energy
                tpower = K.sum(spec)
                # take ratio
                ratio = low_energy / float(tpower)
            else:
                # make exception where fundamental is greater than
                ratio = float(0)

            all_ratio.append(ratio)

            """
            Spectral centroid of the segment
            """
            # spectral centroid
            SC = K.sum(freq * spec) / float(K.sum(spec))
            all_SC.append(SC)

            """
            HF decay
            - linear regression of the values above the warmth region
            """
            above_WR_spec = timbral_util.log10(spec[WR_upper_f_limit_idx:])
            above_WR_freq = timbral_util.log10(freq[WR_upper_f_limit_idx:])
            # tf.ones_like(above_WR_freq)
            metrics = tf.stack([above_WR_freq, tf.ones_like(above_WR_freq)])
            # create a linear regression model
            # TODO make raw regression with matrices
            print(tf.transpose(metrics), above_WR_spec)
            lstsq = tf.linalg.lstsq(tf.transpose(
                metrics), above_WR_spec[:, None])
            y_pred = K.sum(metrics * lstsq, axis=0)
            y_true = above_WR_spec
            # u = ((y_true - y_pred) ** 2).sum()
            u = K.sum((y_true - y_pred) ** 2)
            # v= ((y_true - y_true.mean()) ** 2).sum()
            v = K.sum(((y_true - K.mean(y_true)) ** 2))
            # score = (1-u/v)
            decay_score = 1-u/v
            """
            decay_score = None
            model = linear_model.LinearRegression(
                fit_intercept=False)  # model.coef_ = (2), lstsq = (2,1)
            model.fit(tf.transpose(metrics), above_WR_spec)
            print("scikit model", model.predict(
                tf.transpose(metrics)))
            decay_score = model.score(tf.transpose(metrics), above_WR_spec)
            print("tf csore, sk score", score, decay_score)
            """
            all_decay_score.append(decay_score)

        """
        get mean values
        """
        mean_SC = timbral_util.tf_log10(K.mean(tf.stack(all_SC)))
        mean_decay_score = tf.cast(
            K.mean(tf.stack(all_decay_score)), audio_samples.dtype)
        weighted_mean_ratio = timbral_util.tf_average(
            tf.stack(all_ratio), tf.stack(all_rms))
        """append to different arrays"""
        mean_SC_array.append(mean_SC)
        weighted_hf_array.append(weighted_hf)
        mean_wr_array.append(mean_wr)
        mean_decay_score_array.append(mean_decay_score)
        weighted_mean_ratio_array.append(weighted_mean_ratio)

    mean_SC = tf.stack(mean_SC_array)
    weighted_hf = tf.stack(weighted_hf_array)

    mean_wr = tf.stack(mean_wr_array)

    mean_decay_score = tf.stack(mean_decay_score_array)

    weighted_mean_ratio = tf.stack(weighted_mean_ratio_array)
    print("output", np.array([mean_SC, weighted_hf, mean_wr,
                              mean_decay_score, weighted_mean_ratio]).flatten())
    if dev_output:
        return mean_SC, weighted_hf, mean_wr, mean_decay_score, weighted_mean_ratio
    else:

        """
         Apply regression model
        all_metrics = np.ones(6)
        all_metrics[0] = mean_SC
        all_metrics[1] = weighted_hf
        all_metrics[2] = mean_wr
        all_metrics[3] = mean_decay_score
        all_metrics[4] = weighted_mean_ratio

        coefficients = np.array(
            [
                -4.464258317026696,
                -0.08819320850778556,
                0.29156539973575546,
                17.274733561081554,
                8.403340066029507,
                45.21212125085579,
            ]
        )
        """
        warmth = -4.464258317026696 * mean_SC + -0.08819320850778556 * weighted_hf + 0.29156539973575546 * mean_wr +\
            17.274733561081554 * mean_decay_score + 8.403340066029507 *\
            weighted_mean_ratio + 45.21212125085579

        print("warmth", warmth)
        # clip output between 0 and 100
        if clip_output:
            warmth = timbral_util.output_clip(warmth)

        return warmth
