from __future__ import division
import numpy as np
import soundfile as sf
from . import timbral_util
from scipy.signal import spectrogram
import tensorflow as tf
import tensorflow.keras.backend as K


def timbral_brightness(
    fname,
    fs=0,
    dev_output=False,
    clip_output=False,
    phase_correction=False,
    threshold=0,
    ratio_crossover=2000,
    centroid_crossover=100,
    stepSize=1024,
    blockSize=2048,
    minFreq=20,
    take_first=None,
    in_audio=None
):
    """
      This function calculates the apparent Brightness of an audio file.
      This version of timbral_brightness contains self loudness normalising methods and can accept arrays as an input
      instead of a string filename.

      Version 0.4

      Required parameter
       :param fname:               string or numpy array
                                   string, audio filename to be analysed, including full file path and extension.
                                   numpy array, array of audio samples, requires fs to be set to the sample rate.

      Optional parameters
       :param fs:                  int/float, when fname is a numpy array, this is a required to be the sample rate.
                                   Defaults to 0.
       :param dev_output:          bool, when False return the brightness, when True return all extracted features.
       :param clip_output:         bool, force the output to be between 0 and 100.
       :param phase_correction:    bool, Perform phase checking before summing to mono.
       :param threshold:           Threshold below which to ignore the energy in a time window, default to 0.
       :param ratio_crossover:     Crossover frequency for calculating the HF energy ratio, default to 2000 Hz.
       :param centroid_crossover:  Highpass frequency for calculating the spectral centroid, default to 100 Hz.
       :param stepSize:            Step size for calculating spectrogram, default to 1024.
       :param blockSize:           Block size (fft length) for calculating spectrogram, default to 2048.
       :param minFreq:             Frequency for high-pass filtering audio prior to all analysis, default to 20 Hz.

       :return:                    Apparent brightness of audio file, float.

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
    if fname is None:
        audio_samples = in_audio
    else:
        audio_samples, fs = timbral_util.file_read(
            fname, fs, phase_correction=phase_correction
        )
    if take_first:
        audio_samples = audio_samples[:take_first]
    """
      Filter audio
    """
    # highpass audio at minimum frequency
    audio_samples = timbral_util.filter_audio_highpass(
        audio_samples, crossover=minFreq, fs=fs
    )
    audio_samples = timbral_util.filter_audio_highpass(
        audio_samples, crossover=minFreq, fs=fs
    )
    audio_samples = timbral_util.filter_audio_highpass(
        audio_samples, crossover=minFreq, fs=fs
    )

    # get highpass audio at ratio crossover
    ratio_highpass_audio = timbral_util.filter_audio_highpass(
        audio_samples, ratio_crossover, fs
    )
    ratio_highpass_audio = timbral_util.filter_audio_highpass(
        ratio_highpass_audio, ratio_crossover, fs
    )
    ratio_highpass_audio = timbral_util.filter_audio_highpass(
        ratio_highpass_audio, ratio_crossover, fs
    )

    # get highpass audio at centroid crossover
    centroid_highpass_audio = timbral_util.filter_audio_highpass(
        audio_samples, centroid_crossover, fs
    )
    centroid_highpass_audio = timbral_util.filter_audio_highpass(
        centroid_highpass_audio, centroid_crossover, fs
    )
    centroid_highpass_audio = timbral_util.filter_audio_highpass(
        centroid_highpass_audio, centroid_crossover, fs
    )

    """
     Get spectrograms
    """
    # normalise audio to the maximum value in the unfiltered audio
    ratio_highpass_audio *= 1.0 / max(abs(audio_samples))
    centroid_highpass_audio *= 1.0 / max(abs(audio_samples))
    audio_samples *= 1.0 / max(abs(audio_samples))

    # set FFT parameters
    nfft = blockSize
    hop_size = int(3 * nfft / 4)

    # check that audio is long enough to generate spectrograms
    if len(audio_samples) >= nfft:
        # get spectrogram
        _, _, ratio_all_spec = spectrogram(
            audio_samples,
            fs,
            "hamming",
            nfft,
            hop_size,
            nfft,
            False,
            True,
            "spectrum",
        )
        _, ratio_hp_time, ratio_hp_spec = spectrogram(
            ratio_highpass_audio,
            fs,
            "hamming",
            nfft,
            hop_size,
            nfft,
            False,
            True,
            "spectrum",
        )
        centroid_hp_freq, _, centroid_hp_spec = spectrogram(
            centroid_highpass_audio,
            fs,
            "hamming",
            nfft,
            hop_size,
            nfft,
            False,
            True,
            "spectrum",
        )
    else:
        _, _, ratio_all_spec = spectrogram(
            audio_samples,
            fs,
            "hamming",
            len(audio_samples),
            len(audio_samples) - 1,
            nfft,
            False,
            True,
            "spectrum",
        )
        _, ratio_hp_time, ratio_hp_spec = spectrogram(
            ratio_highpass_audio,
            fs,
            "hamming",
            len(ratio_highpass_audio),
            len(ratio_highpass_audio) - 1,
            nfft,
            False,
            True,
            "spectrum",
        )
        centroid_hp_freq, _, centroid_hp_spec = spectrogram(
            centroid_highpass_audio,
            fs,
            "hamming",
            len(centroid_highpass_audio),
            len(centroid_highpass_audio) - 1,
            nfft,
            False,
            True,
            "spectrum",
        )

    # initialise variables for storing data
    all_ratio = []
    all_hp_centroid = []
    all_tpower = []
    all_hp_centroid_tpower = []

    # set threshold level at zero
    threshold_db = threshold
    if threshold_db == 0:
        threshold = 0
        # hp_threshold = 0
    else:
        max_power = max(np.sum(ratio_all_spec, axis=1))
        threshold = max_power * timbral_util.db2mag(threshold_db)
        # get the threshold for centroid
        # centroid_hp_max_power = max(np.sum(centroid_hp_spec, axis=1))
        # hp_min_power = min(np.sum(hp_spec, axis=1))
        # hp_threshold = hp_max_power * timbral_util.db2mag(threshold_db)
    # threshold = 0.0

    """
      Calculate features for each time window
    """
    for idx in range(len(ratio_hp_time)):  #
        # get the current spectrum for this time window
        current_ratio_hp_spec = ratio_hp_spec[:, idx]
        current_ratio_all_spec = ratio_all_spec[:, idx]
        current_centroid_hp_spec = centroid_hp_spec[:, idx]

        # get the power within each spectrum
        tpower = np.sum(current_ratio_all_spec)
        hp_tpower = np.sum(current_ratio_hp_spec)
        # check there is energy in the time window before calculating the ratio (greater than 0)
        if tpower > threshold:
            # get the ratio
            all_ratio.append(hp_tpower / tpower)
            # store the powef for weighting
            all_tpower.append(tpower)

        # get the tpower to assure greater than zero
        hp_centroid_tpower = np.sum(current_centroid_hp_spec)
        if hp_centroid_tpower > 0.0:
            # get the centroid
            all_hp_centroid.append(
                np.sum(
                    current_centroid_hp_spec
                    * centroid_hp_freq[: len(current_centroid_hp_spec)]
                )
                / np.sum(current_centroid_hp_spec)
            )
            # store the tpower for weighting
            all_hp_centroid_tpower.append(hp_centroid_tpower)

    """
      Get mean and weighted average values
    """
    # mean_ratio = np.mean(all_ratio)
    # mean_hp_centroid = np.mean(all_hp_centroid)

    weighted_mean_ratio = np.average(all_ratio, weights=all_tpower)
    weighted_mean_hp_centroid = np.average(
        all_hp_centroid, weights=all_hp_centroid_tpower
    )

    # np.log10(weighted_mean_ratio), np.log10(weighted_mean_hp_centroid))

    if dev_output:
        # return the ratio and centroid
        return np.log10(weighted_mean_ratio), np.log10(weighted_mean_hp_centroid)
        # return mean_ratio, mean_hp_centroid

    else:
        # perform thye linear regression
        all_metrics = np.ones(3)
        all_metrics[0] = np.log10(weighted_mean_ratio)
        all_metrics[1] = np.log10(weighted_mean_hp_centroid)
        # all_metrics[2] = np.log10(weighted_mean_ratio) * np.log10(weighted_mean_hp_centroid)

        coefficients = np.array(
            [4.613128018020465, 17.378889309312974, 17.434733750553022]
        )

        # coefficients = np.array([-2.9197705625030235, 9.048261758526614, 3.940747859061009, 47.989783427908705])
        bright = np.sum(all_metrics * coefficients)

        if clip_output:
            bright = timbral_util.output_clip(bright)

        return bright


@tf.function(experimental_relax_shapes=False)
def tf_timbral_brightness_2(
    audio_samples,
    fs=0,
    dev_output=False,
    clip_output=False,
    phase_correction=None,
    threshold=None,
    ratio_crossover=2000,
    centroid_crossover=100,
    stepSize=1024,
    blockSize=2048,
    minFreq=20,
    compensated=False,
):
    assert (
        len(audio_samples.get_shape().as_list()) == 3
    ), "Should be BNC at the input. Got {}".format(audio_samples)
    audio_samples = audio_samples[:, :, 0]
    # highpass audio at minimum frequency
    audio_samples = timbral_util.tf_filter_audio_highpass(
        audio_samples, crossover=minFreq, fs=fs, order=3
    )
    audio_samples = timbral_util.tf_filter_audio_highpass(
        audio_samples, crossover=minFreq, fs=fs, order=3
    )

    """
    audio_samples = timbral_util.tf_filter_audio_highpass(
        audio_samples, crossover=minFreq, fs=fs)
    audio_samples = timbral_util.tf_filter_audio_highpass(
        audio_samples, crossover=minFreq, fs=fs)
    """
    # get highpass audio at ratio crossover
    ratio_highpass_audio = timbral_util.tf_filter_audio_highpass(
        audio_samples, ratio_crossover, fs, order=3
    )
    ratio_highpass_audio = timbral_util.tf_filter_audio_highpass(
        ratio_highpass_audio, ratio_crossover, fs, order=3
    )
    """
    ratio_highpass_audio = timbral_util.tf_filter_audio_highpass(
        ratio_highpass_audio, ratio_crossover, fs)
    """
    # get highpass audio at centroid crossover
    centroid_highpass_audio = timbral_util.tf_filter_audio_highpass(
        audio_samples, centroid_crossover, fs, order=3
    )
    centroid_highpass_audio = timbral_util.tf_filter_audio_highpass(
        centroid_highpass_audio, centroid_crossover, fs, order=3
    )
    """
    centroid_highpass_audio = timbral_util.tf_filter_audio_highpass(
        centroid_highpass_audio, centroid_crossover, fs)
    centroid_highpass_audio = timbral_util.tf_filter_audio_highpass(
        centroid_highpass_audio, centroid_crossover, fs)
        
    """

    """
    Get spectrograms
    """
    # normalise audio to the maximum value in the unfiltered audio
    max_val = 1.0 / K.max(K.abs(audio_samples), axis=[1], keepdims=True)

    ratio_highpass_audio = ratio_highpass_audio * max_val
    centroid_highpass_audio = centroid_highpass_audio * max_val
    audio_samples = audio_samples * max_val

    # set FFT parameters
    nfft = blockSize
    hop_size = int(3 * nfft / 4)

    # check that audio is long enough to generate spectrograms
    if audio_samples.shape[1] >= nfft:
        # get spectrogram
        _, _, ratio_all_spec = timbral_util.compat_spectrogram(
            audio_samples,
            fs,
            "hamming",
            nfft,
            hop_size,
            nfft,
            False,
            True,
            "spectrum",
        )
        _, ratio_hp_time, ratio_hp_spec = timbral_util.compat_spectrogram(
            ratio_highpass_audio,
            fs,
            "hamming",
            nfft,
            hop_size,
            nfft,
            False,
            True,
            "spectrum",
        )
        centroid_hp_freq, _, centroid_hp_spec = timbral_util.compat_spectrogram(
            centroid_highpass_audio,
            fs,
            "hamming",
            nfft,
            hop_size,
            nfft,
            False,
            True,
            "spectrum",
        )
    else:
        NotImplemented
        _, _, ratio_all_spec = timbral_util.compat_spectrogram(
            audio_samples,
            fs,
            "hamming",
            len(audio_samples),
            len(audio_samples) - 1,
            nfft,
            False,
            True,
            "spectrum",
        )
        _, _, ratio_hp_spec = timbral_util.compat_spectrogram(
            ratio_highpass_audio,
            fs,
            "hamming",
            len(ratio_highpass_audio),
            len(ratio_highpass_audio) - 1,
            nfft,
            False,
            True,
            "spectrum",
        )
        centroid_hp_freq, _, centroid_hp_spec = timbral_util.compat_spectrogram(
            centroid_highpass_audio,
            fs,
            "hamming",
            len(centroid_highpass_audio),
            len(centroid_highpass_audio) - 1,
            nfft,
            False,
            True,
            "spectrum",
        )
    # print(ratio_all_time.shape, ratio_hp_time.shape, centroid_hp_time.shape)
    # initialise variables for storing data
    # set threshold level at zero
    # threshold in an ununsed parameter
    threshold = 0.0

    """
    Calculate features for each time window
    """
    if threshold == 0.0:
        # get the current spectrum for this time window
        """
        Original :
        current_ratio_hp_spec = ratio_hp_spec[:, idx]
        current_ratio_all_spec = ratio_all_spec[:, idx]
        current_centroid_hp_spec = centroid_hp_spec[:, idx]
        Note :
        compat_spectrogram sort le spectrogramme "?? l'envers". 
        On aura current_centroid_hp_spec = centroid_hp_spec[idx, :] en fait
        Cela induit l'utilisation de axis 0 (ou 1 ?)
        """
        # get the power within each spectrum
        tpower = K.sum(ratio_all_spec, axis=-1) + K.epsilon()
        hp_tpower = K.sum(ratio_hp_spec, axis=-1)
        # check there is energy in the time window before calculating the ratio (greater than 0)
        """
        if tpower > threshold:
            # get the ratio
            all_ratio.append(hp_tpower / tpower)
            # store the powef for weighting
            all_tpower.append(tpower)
        """
        # Threshold is always 0. : the if is useless
        # get the ratio

        all_ratio = hp_tpower / tpower

        # store the powef for weighting
        all_tpower = tpower
        # get the tpower to assure greater than zero
        hp_centroid_tpower = K.sum(centroid_hp_spec, axis=-1)

        # if hp_centroid_tpower > 0.0:
        # get the centroid
        """
        all_hp_centroid.append(np.sum(current_centroid_hp_spec * centroid_hp_freq[:len(current_centroid_hp_spec)]) /
                        np.sum(current_centroid_hp_spec))
        """
        # print("centroid_hp_spec", centroid_hp_spec)
        # print("centroid_hp_freq", centroid_hp_freq)
        all_hp_centroid = tf.reduce_sum(
            centroid_hp_spec * centroid_hp_freq[None, None, :], axis=-1
        ) / tf.reduce_sum(centroid_hp_spec + K.epsilon(), axis=-1)

        # store the tpower for weighting
        all_hp_centroid_tpower = hp_centroid_tpower

    else:
        raise NotImplementedError
    """
    Get mean and weighted average values
    """
    # mean_ratio = K.mean(all_ratio, axis=-1)

    # mean_hp_centroid = K.mean(all_hp_centroid, axis=-1)

    # Diff??rence pas bien ??lev??e :: job done.

    # TODO :: operation should be batch compatible.
    """
    weighted_mean_ratio = K.sum(
        all_ratio * all_tpower, axis=-1)/K.sum(all_tpower, axis=-1)
    weighted_mean_hp_centroid = K.sum(
        all_hp_centroid * all_hp_centroid_tpower, axis=-1)/K.sum(all_hp_centroid_tpower, axis=-1)
    """
    weighted_mean_ratio = timbral_util.tf_average(
        all_ratio, all_tpower, epsilon=None)
    weighted_mean_hp_centroid = timbral_util.tf_average(
        all_hp_centroid, all_hp_centroid_tpower, epsilon=None
    )
    # print("all_ratio", weighted_mean_ratio, weighted_mean_hp_centroid)

    base = tf.cast(tf.math.log(10.0), weighted_mean_ratio.dtype)
    # print("scores : ", tf.math.log(weighted_mean_ratio)/base, tf.math.log(weighted_mean_hp_centroid)/base )
    if dev_output:
        # return the ratio and centroid
        return (
            tf.math.log(weighted_mean_ratio) / base,
            tf.math.log(weighted_mean_hp_centroid) / base,
        )
        # eturn mean_ratio, mean_hp_centroid
    else:
        # perform thye linear regression
        """
   all_metrics = np.ones(3)
        all_metrics[0] = np.log10(weighted_mean_ratio)
        all_metrics[1] = np.log10(weighted_mean_hp_centroid)
        # all_metrics[2] = np.log10(weighted_mean_ratio) * np.log10(weighted_mean_hp_centroid)

        coefficients = np.array(
            [4.613128018020465, 17.378889309312974, 17.434733750553022])

        # coefficients = np.array([-2.9197705625030235, 9.048261758526614, 3.940747859061009, 47.989783427908705])
        bright = np.sum(all_metrics * coefficients)

        if clip_output:
            bright = timbral_util.output_clip(bright)
        """

        # all_metrics[2] = np.log10(weighted_mean_ratio) * np.log10(weighted_mean_hp_centroid)

        # coefficients = tf.constant([4.613128018020465, 17.378889309312974, 17.434733750553022], dtype=all_metrics.dtype)

        # coefficients = np.array([-2.9197705625030235, 9.048261758526614, 3.940747859061009, 47.989783427908705])
        # K.sum seems to give None gradients...
        #   17.434733750553022
        # bright = K.sum(all_metrics * coefficients[:,None])
        if not compensated:
            output = (
                4.613128018020465 * tf.math.log(weighted_mean_ratio) / base
                + 17.378889309312974 *
                tf.math.log(weighted_mean_hp_centroid) / base
                + 17.434733750553022 * tf.ones_like(weighted_mean_hp_centroid)
            )
        else:
            # linear reg -> [ 4.5712724 17.355247 ] 17.197018
            output = (
                4.5712724 * tf.math.log(weighted_mean_ratio) / base
                + 17.355247 * tf.math.log(weighted_mean_hp_centroid) / base
                + 17.197018 * tf.ones_like(weighted_mean_hp_centroid)
            )
        # print(tf.math.log(weighted_mean_ratio)/base,              tf.math.log(weighted_mean_hp_centroid)/base)

        if clip_output:
            output = timbral_util.output_clip(output)
        return output
