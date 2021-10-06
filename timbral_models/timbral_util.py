from __future__ import division, print_function
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter, spectrogram, freqz
import scipy.stats
import pyloudnorm as pyln
import six
import tensorflow as tf
import tensorflow.keras.backend as K

"""
  The timbral util is a collection of functions that can be accessed by the individual timbral models.  These can be
  used for extracting features or manipulating the audio that are useful to multiple attributes.

  Version 0.4

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

FRAME_LENGTH = 4096
OVERLAP = 0.75


def print_yellow(*args):
    print("\033[93m", *args, "\033[0m")


def print_warning(*args):
    print_yellow(*args)


def print_blue(*args):
    print("\033[94m", *args, "\033[0m")


def print_green(*args):
    print("\033[92m", *args, "\033[0m")


def print_fail(*args):
    print("\033[91m", *args, "\033[0m")


"""
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'

"""


def freqz_(*args, **kwargs):
    _, h = freqz(*args, **kwargs)
    return h


def log10(x):
    return tf.math.log(x) / tf.math.log(tf.cast(10.0, x.dtype))


def get_frequency_list(stft, fs=44100):
    n_bins = stft.shape[-1]
    return tf.linspace(tf.cast(0.0, stft.dtype), 0.5 * fs, n_bins)


def tf_freqz(b, a, n_bins):
    frequencies = tf.math.exp(
        1j * tf.linspace(1j * 0.0, 2 * np.pi, n_bins, name="frequencies")
    )
    response = tf.math.polyval(b, frequencies) / \
        tf.math.polyval(a, frequencies)
    return response


def tf_average(x, y, axis=-1, epsilon=K.epsilon(), debug_check=False):
    """
    Convenience function, equivalent to np.average 
    Arguments :
    x,y : tf.Tensor. y is the tensor containing the weights
    axis = axis to iterate over. W=arning : Default to -1
    espilon : small number to avoid dividing by 0. Default to keras default (1e-7)
    """
    if epsilon is None:
        out = K.sum(x * y, axis=axis) / K.sum(y, axis=axis)
    else:
        out = K.sum(x * y, axis=axis)
        out = out / (K.sum(y, axis=axis) + epsilon)

    if debug_check:
        tf.debugging.check_numerics(
            out, "nan or inf in tf_average :: {}".format(out))
    return out


def tf_filter(
    b,
    a,
    signals,
    frame_length=FRAME_LENGTH,
    frame_step=int(FRAME_LENGTH * (1 - OVERLAP)),
):
    # format is fixed::  b,n
    # frame_length =  # Should be a power of 2 to have correct nbins
    # STFT sur le canal gauche (à voir pour une somme)

    length = signals.shape[1]
    paddings = tf.constant([[0, 0], [frame_length - frame_step, 0]])
    signals = tf.pad(signals, paddings, "CONSTANT")

    stft = tf.signal.stft(signals, frame_length, frame_step, pad_end=True)
    n_bins = stft.shape[-1]

    # n_bins = stft.shape[-1]
    h = tf.numpy_function(freqz_, [b, a, n_bins], tf.complex128, name="freqz")
    # h = freqz_(b,a,n_bins)
    # h = tf_freqz(b,a,n_bins)
    #    freqz(b, a, n_bins)
    # f is 0-0.5 with 1 being fs. Should check with scipy butter for full work
    y_freq = stft * tf.cast(tf.reshape(h, [1, 1, n_bins]), stft.dtype)
    # print_yellow("tf_filter::y_freq", y_freq.shape)
    # print_yellow("tf_filter::stft", stft.shape)

    # Resynthetise y_freq to y_t (or signals ?)
    # y_freq is "None" in tf.function mode
    inverse_stft = tf.signal.inverse_stft(
        y_freq,
        frame_length,
        frame_step,
        window_fn=tf.signal.inverse_stft_window_fn(frame_step),
    )
    # TODO :: invalid format for batch use
    out = inverse_stft[
        :, frame_length - frame_step: length + (frame_length - frame_step)
    ]
    # print_yellow("tf_filter::inverse_stft out shape", out.shape)
    return out


def db2mag(dB):
    """
      Converts from dB to linear magnitude.

    :param dB:  dB level to be converted.
    :return:    linear magnitude of the dB input.
    """
    mag = 10 ** (dB / 20.0)
    return mag


def get_percussive_audio(audio_samples, return_ratio=True):
    """
      Gets the percussive comonent of the audio file.
      Currently, the default values for harmonic/percussive decomposition have been used.
      Future updates may change the defaults for better separation or to improve the correlation to subjective data.

    :param audio_samples:   The audio samples to be harmonicall/percussively separated
    :param return_ratio:    Determins the value returned by the function.

    :return:                If return_ratio is True (default), the ratio of percussive energy is returned.
                            If False, the function returns the percussive audio as a time domain array.
    """
    # TODO : convert to tensorflow
    # use librosa decomposition
    D = librosa.core.stft(audio_samples)
    H, P = librosa.decompose.hpss(D)

    # inverse transform to get time domain arrays
    percussive_audio = librosa.core.istft(P)
    harmonic_audio = librosa.core.istft(H)

    if return_ratio:
        # frame by frame RMS energy
        percussive_energy = calculate_rms_enveope(
            percussive_audio, step_size=1024, overlap_step=512, normalise=False
        )
        harmonic_energy = calculate_rms_enveope(
            harmonic_audio, step_size=1024, overlap_step=512, normalise=False
        )

        # set defaults for storing the data
        ratio = []
        t_power = []

        # get the ratio for each RMS time frame
        for i in range(len(percussive_energy)):
            if percussive_energy[i] != 0 or harmonic_energy[i] != 0:
                # if percussive_energy[i] != 0 and harmonic_energy[i] != 0:
                ratio.append(
                    percussive_energy[i] /
                    (percussive_energy[i] + harmonic_energy[i])
                )
                t_power.append((percussive_energy[i] + harmonic_energy[i]))

        if t_power:
            # take a weighted average of the ratio
            ratio = np.average(ratio, weights=t_power)
            return ratio
    else:
        # return the percussive audio when return_ratio is False
        return percussive_audio


def filter_audio_highpass(audio_samples, crossover, fs, order=2):
    """ Calculate and apply a high-pass filter, with a -3dB point of crossover.

    :param audio_samples:   data to be filtered as an array.
    :param crossover:       the crossover frequency of the filter.
    :param fs:              the sampling frequency of the audio file.
    :param order:           order of the filter, defaults to 2.

    :return:                filtered array.
    """
    # TODO : convert to tensorflow

    nyq = 0.5 * fs
    xfreq = crossover / nyq
    b, a = butter(order, xfreq, "high")
    y = lfilter(b, a, audio_samples)
    return y


def high_butter(order, xfreq):
    return butter(order, xfreq, "high")


def low_butter(order, xfreq):
    return butter(order, xfreq, "low")


# @tf.function


def tf_filter_audio_highpass(
    audio_samples,
    crossover,
    fs,
    order=2,
    frame_length=FRAME_LENGTH,
    frame_step=int(FRAME_LENGTH * (1 - OVERLAP)),
):
    """ Calculate and apply a high-pass filter, with a -3dB point of crossover.

    :param audio_samples:   data to be filtered as an array.
    :param crossover:       the crossover frequency of the filter.
    :param fs:              the sampling frequency of the audio file.
    :param order:           order of the filter, defaults to 2.

    :return:                filtered array.
    """
    # TODO : convert to tensorflow
    print_yellow(
        "Tracing tf_filter_audio_highpass with",
        audio_samples.shape,
        crossover,
        order,
        frame_length,
        frame_step,
    )

    nyq = 0.5 * fs
    xfreq = crossover / nyq
    # b, a = butter(order, xfreq, 'high')
    out = tf.numpy_function(
        high_butter, [order, xfreq], (tf.float64,
                                      tf.float64), name="high_butter"
    )
    b, a = out[0], out[1]

    y = tf_filter(b, a, audio_samples, frame_length, frame_step)
    return y


def filter_audio_lowpass(audio_samples, crossover, fs, order=2):
    """ Calculate and apply a low-pass filter, with a -3dB point of crossover.

    :param audio_samples:   data to be filtered as an array.
    :param crossover:       the crossover frequency of the filter.
    :param fs:              the sampling frequency of the audio file.
    :param order:           order of the filter, defaults to 2.

    :return:                filtered array.
    """
    # TODO : convert to tensorflow
    nyq = 0.5 * fs
    xfreq = crossover / nyq
    b, a = butter(order, xfreq, "low")
    y = lfilter(b, a, audio_samples)

    return y


def tf_filter_audio_lowpass(
    audio_samples, crossover, fs, order=2, frame_length=FRAME_LENGTH, frame_step=64
):
    """ Calculate and apply a low-pass filter, with a -3dB point of crossover.

    :param audio_samples:   data to be filtered as an array.
    :param crossover:       the crossover frequency of the filter.
    :param fs:              the sampling frequency of the audio file.
    :param order:           order of the filter, defaults to 2.

    :return:                filtered array.
    """
    # TODO : convert to tensorflow
    nyq = 0.5 * fs
    xfreq = crossover / nyq
    # b, a = butter(order, xfreq, 'low')
    out = tf.numpy_function(
        low_butter, [order, xfreq], (tf.float64, tf.float64), name="low_butter"
    )
    b, a = out[0], out[1]
    # y = lfilter(b, a, audio_samples)
    y = tf_filter(b, a, audio_samples, frame_length, frame_step)
    return y


def butter_bandpass(lowcut, highcut, fs, order=2):
    """ Design a butterworth bandpass filter """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def filter_audio_bandpass(audio_samples, f0, noct, fs, order=2):
    """ Calculate and apply an n/octave butterworth bandpass filter, centred at f0 Hz.

    :param audio_samples: the audio file as an array
    :param fs: the sampling frequency of the audio file
    :param f0: the centre frequency of the bandpass filter
    :param bandwidth: the bandwidth of the filter
    :param order: order of the filter, defaults to 2

    :return: audio file filtered
    """
    # TODO : convert to tensorflow

    fd = 2 ** (1.0 / (noct * 2))
    lowcut = f0 / fd
    highcut = f0 * fd

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, audio_samples)
    return y


def tf_filter_audio_bandpass(
    audio_samples, f0, noct, fs, order=2, frame_length=FRAME_LENGTH, frame_step=64
):
    """ Calculate and apply a low-pass filter, with a -3dB point of crossover.

    :param audio_samples:   data to be filtered as an array.
    :param crossover:       the crossover frequency of the filter.
    :param fs:              the sampling frequency of the audio file.
    :param order:           order of the filter, defaults to 2.

    :return:                filtered array.
    """
    # TODO : convert to tensorflow
    fd = 2 ** (1.0 / (noct * 2))
    lowcut = f0 / fd
    highcut = f0 * fd

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # y = lfilter(b, a, audio_samples)
    y = tf_filter(b, a, audio_samples, frame_length, frame_step)
    return y


def compat_spectrogram(
    x,
    fs=1.0,
    window="hamming",
    nperseg=None,
    noverlap=None,
    nfft=None,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
):
    """ spectrogram(audio_samples, fs, 'hamming', nfft, hop_size, nfft, 'constant', True, 'spectrum')
                                                                  """
    del return_onesided, axis, mode
    # detrend detrends on the spectrogram on a "per line" basis.
    # "constant" means normalized by mean, "linear" with least-squere regression.
    # TODO :: implement detrending for (almost) full compatibility with scipy
    if window == "hamming":
        window = tf.signal.hamming_window
    elif window == "hann":
        window = tf.signal.hann_window
    else:
        raise ValueError(
            "Window name unknown :: got {} expected 'hamming' of 'hann'".format(
                window)
        )
    if scaling.lower() == "density":
        scale = tf.reduce_sum(window(nfft, dtype=x.dtype)) * tf.sqrt(fs)
    elif scaling.lower() == "none":
        scale = 1.
    else:
        scale = tf.reduce_sum(window(nfft, dtype=x.dtype))
    # noverlap est le nombre de points communs entre deux fenêtres pour scipy
    # Tensorflow prend le nombre de points entre le début des fenêtres
    hop_size = nperseg - noverlap
    if len(x.get_shape().as_list()) == 3:
        spectrogram = tf.signal.stft(
            x[:, :, 0],
            frame_length=nperseg,
            frame_step=hop_size,
            fft_length=None,
            window_fn=tf.signal.hamming_window,
            pad_end=False,
            name=None,
        )
    elif len(x.get_shape().as_list()) == 2:
        spectrogram = tf.signal.stft(
            x[:, :],
            frame_length=nperseg,
            frame_step=hop_size,
            fft_length=None,
            window_fn=tf.signal.hamming_window,
            pad_end=False,
            name=None,
        )
    else:
        raise ValueError(
            "Rank is not ok in compat_spectrogram expected 2 or 3, got {} with {}".format(
                len(x.get_shape().as_list()), x
            )
        )
    """
    N = np.sum(tf.signal.hamming_window(
    nfft, periodic=True, dtype=tf.dtypes.float32, name=None
))
calc = 2*(np.abs(Sxx_t.numpy())/N)**2
    """
    # spectrum :: scale = 1.0 / win.sum()**2
    # density :: scale = 1.0 / (fs * (win*win).sum())
    if scaling.lower() == "none":
        spectrogram = tf.abs(spectrogram)
    else:
        spectrogram = 2 * (tf.abs(spectrogram) / scale)**2
    f = get_frequency_list(spectrogram, fs=fs)
    sss = spectrogram.get_shape().as_list()[-2]
    """
    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                     nperseg - noverlap)/float(fs)
    
    t = (tf.linspace(tf.cast(0.0, spectrogram.dtype),
                     sss - 1, sss) + 1) * noverlap / fs
    """
    t = tf.range(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                 nperseg - noverlap)/float(fs)
    # 5/10/20 :: f validé
    # 5/10/20 :: t validé
    # scipy spectrogram use last column as time :: needs transpotion of last two columns.
    # print(spectrogram)
    if detrend == "constant":
        NotImplemented
    elif detrend == "linear":
        NotImplemented

    return f, t, spectrogram


def return_loop(
    onset_loc,
    envelope,
    function_time_thresh,
    hist_threshold,
    hist_time_samples,
    nperseg=512,
):
    """ This function is used by the calculate_onsets method.
     This looks backwards in time from the attack time and attempts to find the exact onset point by
     identifying the point backwards in time where the envelope no longer falls.
     This function includes a hyteresis to account for small deviations in the attack due to the
     envelope calculation.

     Function looks 10ms (function_time_thresh) backwards from the onset time (onset_loc), looking for any sample
     lower than the current sample.  This repeats, starting at the minimum value until no smaller value is found.
     Then the function looks backwards over 200ms, checking if the increase is greater than 10% of the full envelope's
     dynamic range.

        onset_loc:              The onset location estimated by librosa (converted to time domain index)
        envelope:               Envelope of the audio file
        function_time_thresh:   Time threshold for looking backwards in time.  Set in the timbral_hardness code
                                to be the number of samples that equates to 10ms
        hist_threshold:         Level threshold to check over 200ms if the peak is small enough to continue looking
                                backwards in time.
        hist_time_samples:      Number of samples to look back after finding the minimum value over 10ms, set to 200ms.
    """

    # define flag for exiting while loop
    found_start = False

    while not found_start:
        # get the current sample value
        current_sample = envelope[int(onset_loc)]
        # get the previous 10ms worth of samples
        if onset_loc - function_time_thresh > 0:
            evaluation_array = envelope[
                onset_loc - function_time_thresh - 1: onset_loc
            ]
        else:
            evaluation_array = envelope[: onset_loc - 1]

        if min(evaluation_array) - current_sample <= 0:
            """
             If the minimum value within previous 10ms is less than current sample,
             move to the start position to the minimum value and look again.
            """
            min_idx = np.argmin(evaluation_array)
            new_onset_loc = min_idx + onset_loc - function_time_thresh - 1

            if new_onset_loc > nperseg:
                onset_loc = new_onset_loc
            else:
                """ Current index is close to start of the envelope, so exit with the idx as 512 """
                return 0

        else:
            """
             If the minimum value within previous 10ms is greater than current sample,
             introduce the time and level hysteresis to check again.
            """
            # get the array of 200ms previous to the current onset idx
            if (onset_loc - hist_time_samples - 1) > 0:
                hyst_evaluation_array = envelope[
                    onset_loc - hist_time_samples - 1: onset_loc
                ]
            else:
                hyst_evaluation_array = envelope[:onset_loc]

            # values less than current sample
            all_match = np.where(hyst_evaluation_array < envelope[onset_loc])

            # if no minimum was found within the extended time, exit with current onset idx
            if len(all_match[0]) == 0:
                return onset_loc

            # get the idx of the closest value which is lower than the current onset idx
            last_min = all_match[0][-1]
            last_idx = int(onset_loc - len(hyst_evaluation_array) + last_min)

            # get the dynamic range of this segment
            segment_dynamic_range = max(hyst_evaluation_array[last_min:]) - min(
                hyst_evaluation_array[last_min:]
            )

            # compare this dynamic range against the hyteresis threshold
            if segment_dynamic_range >= hist_threshold:
                """
                 The dynamic range is greater than the threshold, therefore this is a separate audio event.
                 Return the current onset idx.
                """
                return onset_loc
            else:
                """
                 The dynamic range is less than the threshold, therefore this is not a separate audio event.
                 Set current onset idx to minimum value and repeat.
                """
                if last_idx >= nperseg:
                    onset_loc = last_idx
                else:
                    """
                     The hysteresis check puts the new threshold too close to the start
                    """
                    return 0


def sample_and_hold_envelope_calculation(
    audio_samples, fs, decay_time=0.2, hold_time=0.01
):
    """
     Calculates the envelope of audio_samples with a 'sample and hold' style function.
     This ensures that the minimum attack time is not limited by low-pass filtering,
     a common method of obtaining the envelope.

    :param audio_samples:   audio array
    :param fs:              sampling frequency
    :param decay_time:      decay time after peak hold
    :param hold_time:       hold time when identifying a decay

    :return:                envelope of audio_samples
    """
    # rectify the audio signal
    abs_samples = abs(audio_samples)
    envelope = []

    # set parameters for envelope function
    # decay rate relative to peak level of audio signal
    decay = max(abs_samples) / (decay_time * fs)
    hold_samples = hold_time * fs  # number of samples to hold before decay
    hold_counter = 0
    previous_sample = 0.0

    # perform the sample, hold, and decay function to obtain envelope
    for sample in abs_samples:
        if sample >= previous_sample:
            envelope.append(sample)
            previous_sample = sample
            hold_counter = 0
        else:
            # check hold length
            if hold_counter < hold_samples:
                hold_counter += 1
                envelope.append(previous_sample)
            else:
                out = previous_sample - decay
                if out > sample:
                    envelope.append(out)
                    previous_sample = out
                else:
                    envelope.append(sample)
                    previous_sample = sample

    # convert to numpy array
    return np.array(envelope)


def tf_sample_and_hold_envelope_calculation(
    audio_samples, fs, decay_time=0.2, hold_time=0.01
):
    """
     Calculates the envelope of audio_samples with a 'sample and hold' style function.
     This ensures that the minimum attack time is not limited by low-pass filtering,
     a common method of obtaining the envelope.

    :param audio_samples:   audio array
    :param fs:              sampling frequency
    :param decay_time:      decay time after peak hold
    :param hold_time:       hold time when identifying a decay

    :return:                envelope of audio_samples
    """
    # rectify the audio signal
    abs_samples = tf.abs(audio_samples)
    envelope = np.zeros(audio_samples.get_shape().as_list())
    # tensor of format BN(C)

    # set parameters for envelope function
    # decay rate relative to peak level of audio signal
    decay = K.max(abs_samples, axis=-1) / (decay_time * fs)
    hold_samples = hold_time * fs  # number of samples to hold before decay
    hold_counter = 0
    previous_sample = 0.0

    # perform the sample, hold, and decay function to obtain envelope
    # function in eager mode
    # b, n = audio_samples.get_shape().as_list()
    for i, elem in enumerate(abs_samples):
        for j, sample in enumerate(elem):
            if sample >= previous_sample:
                envelope[i, j] = sample
                previous_sample = sample
                hold_counter = 0
            else:
                # check hold length
                if hold_counter < hold_samples:
                    hold_counter += 1
                    envelope[i, j] = previous_sample
                else:
                    # the hold length is passed
                    out = previous_sample - decay[i]
                    if out > sample:
                        envelope[i, j] = out
                        previous_sample = out
                    else:
                        envelope[i, j] = sample
                        previous_sample = sample

    # convert to numpy array
    return envelope


def get_spectral_features(
    audio,
    fs,
    lf_limit=20,
    scale="hz",
    cref=27.5,
    power=2,
    window_type="none",
    rollon_thresh=0.05,
):
    del power
    """
     This function calculates the spectral centroid and spectral spread of an audio array.

     :param audio:      Audio array
     :param fs:         Sample rate of audio file
     :param lf_limit:   Low frequency limit, in Hz, to be analysed.  Defaults to 20Hz.
     :param scale:      The frequency scale that calculations should be made over.  if no argument is given, this
                        defaults to 'hz', representing a linear frequency scale.  Options are 'hz', 'mel', 'erb',
                        or 'cents'.
     :param cref:       The reference frequency for calculating cents.  Defaults to 27.5Hz.
     :param power:      The power to raise devaition from specteal centroid, defaults to 2.

     :return:           Returns the spectral centroid, spectral spread, and unitless centroid.
    """
    # use a hanning window
    if window_type == "hann":
        window = np.hanning(len(audio))
    elif window_type == "none":
        window = np.ones(len(audio))
    else:
        raise ValueError("Window type must be set to either 'hann' or 'none'")

    next_pow_2 = int(pow(2, np.ceil(np.log2(len(window)))))
    # get frequency domain representation
    spectrum = np.fft.fft((window * audio), next_pow_2)
    spectrum = np.absolute(spectrum[0: int(len(spectrum) / 2) + 1])

    tpower = np.sum(spectrum)

    if tpower > 0:
        freq = np.arange(0, len(spectrum), 1) * \
            (fs / (2.0 * (len(spectrum) - 1)))

        # find lowest frequency index, zeros used to unpack result
        lf_limit_idx = np.where(freq >= lf_limit)[0][0]
        spectrum = spectrum[lf_limit_idx:]
        freq = freq[lf_limit_idx:]

        # convert frequency to desired frequency scale
        if scale == "hz":
            freq = freq
        elif scale == "mel":
            freq = 1127.0 * np.log(1 + (freq / 700.0))
        elif scale == "erb":
            freq = 21.4 * np.log10(1 + (0.00437 * freq))
        elif freq == "cents":
            # for cents, a
            freq = 1200.0 * np.log2((freq / cref) + 1.0)
        else:
            raise ValueError(
                "Frequency scale type not recognised.  Please use 'hz', 'mel', 'erb', or 'cents'."
            )

        # calculate centroid and spread
        centroid = sum(spectrum * freq) / float(sum(spectrum))

        # old calculation of spread
        deviation = np.abs(freq - centroid)
        spread = np.sqrt(
            np.sum((deviation ** 2) * spectrum) / np.sum(spectrum))

        # new calculation of spread according to librosa
        # spread = np.sqrt(np.sum(spectrum * (deviation ** power))) #** (1. / power))

        cumulative_spectral_power = spectrum[0]
        counter = 0
        rollon_threshold = np.sum(spectrum) * rollon_thresh
        while cumulative_spectral_power < rollon_threshold:
            counter += 1
            cumulative_spectral_power = np.sum(spectrum[:counter])

        if counter == 0:
            counter = 1

        rollon_frequency = freq[counter]
        unitless_centroid = centroid / rollon_frequency

        return centroid, spread, unitless_centroid
    else:
        return 0


def tf_get_spectral_features(
    audio,
    fs,
    lf_limit=20,
    scale="hz",
    cref=27.5,
    power=2,
    window_type="none",
    rollon_thresh=0.05,
):
    del power
    """
     This function calculates the spectral centroid and spectral spread of an audio array.

     :param audio:      Audio tensor, format BN(C)
     :param fs:         Sample rate of audio file
     :param lf_limit:   Low frequency limit, in Hz, to be analysed.  Defaults to 20Hz.
     :param scale:      The frequency scale that calculations should be made over.  if no argument is given, this
                        defaults to 'hz', representing a linear frequency scale.  Options are 'hz', 'mel', 'erb',
                        or 'cents'.
     :param cref:       The reference frequency for calculating cents.  Defaults to 27.5Hz.
     :param power:      The power to raise devaition from specteal centroid, defaults to 2.

     :return:           Returns the spectral centroid, spectral spread, and unitless centroid.
    """
    assert (
        tf.rank(audio) == 2 or tf.rank(audio) == 3
    ), "Tensor should be of rank 2 or 3 (BNC format), got {}".format(tf.rank(audio))
    # use a hanning window
    if window_type == "hann":
        window = tf.signal.hann_window(audio.shape[1])
    elif window_type == "none":
        window = tf.ones(audio.shape)
    else:
        raise ValueError("Window type must be set to either 'hann' or 'none'")

    next_pow_2 = int(pow(2, np.ceil(np.log2(len(window.shape[1])))))
    # get frequency domain representation
    # This should be handled by tf.signal easily....
    """
    spectrum = np.fft.fft((window * audio), next_pow_2)
    spectrum = np.absolute(spectrum[0:int(len(spectrum) / 2) + 1])
    """
    if tf.rank(audio) == 3:
        spectrum = tf.abs(tf.signal.rfft(audio[:, :, 0], next_pow_2))
    else:
        spectrum = tf.abs(tf.signal.rfft(audio[:, :], next_pow_2))

    # tpower = K.sum(spectrum, axis=-1)

    freq = K.arange(0, spectrum.shape[1], 1) * \
        (fs / (2.0 * (spectrum.shape[1] - 1)))

    # find lowest frequency index, zeros used to unpack result
    lf_limit_idx = np.where(freq >= lf_limit)[0][0]
    # TODO :: get lf_limit through maths.
    spectrum = spectrum[:, lf_limit_idx:]
    freq = freq[:, lf_limit_idx:]

    # convert frequency to desired frequency scale
    if scale == "hz":
        freq = freq
    elif scale == "mel":
        freq = 1127.0 * np.log(1 + (freq / 700.0))
    elif scale == "erb":
        freq = 21.4 * np.log10(1 + (0.00437 * freq))
    elif freq == "cents":
        # for cents, a
        freq = 1200.0 * np.log2((freq / cref) + 1.0)
    else:
        raise ValueError(
            "Frequency scale type not recognised.  Please use 'hz', 'mel', 'erb', or 'cents'."
        )

    # calculate centroid and spread
    centroid = K.sum(spectrum * freq, axis=-1) / K.sum(spectrum, axis=-1)

    # old calculation of spread
    deviation = K.abs(freq - centroid)
    spread = K.sqrt(
        K.sum((deviation ** 2) * spectrum, axis=-1) / K.sum(spectrum, axis=-1)
    )

    # new calculation of spread according to librosa
    # spread = np.sqrt(np.sum(spectrum * (deviation ** power))) #** (1. / power))

    cumulative_spectral_power = spectrum[0]
    counter = 0
    rollon_threshold = np.sum(spectrum) * rollon_thresh
    while cumulative_spectral_power < rollon_threshold:
        counter += 1
        cumulative_spectral_power = np.sum(spectrum[:counter])

    if counter == 0:
        counter = 1

    rollon_frequency = freq[counter]
    unitless_centroid = centroid / rollon_frequency

    return centroid, spread, unitless_centroid


def calculate_attack_time(
    envelope_samples,
    fs,
    calculate_attack_segment=True,
    thresh_no=8,
    normalise=True,
    m=3,
    calculation_type="min_effort",
    gradient_calulation_type="all",
    return_descriptive_data=False,
    max_attack_time=-1,
):
    """
      Calculate the attack time from the envelope of a signal.

    Required inputs
    :param envelope_samples:            envelope of the audio file, suggested to be calculated with
                                        sample_and_hold_envelope_calculation.
    :param fs:                          sample rate of the envelope_samples.

    Optional inputs
    :param calculate_attack_segment:    If the attack segment of the onset should be calculated before estimating the
                                        attack time. bool, default to True.
    :param thresh_no:                   Number of thresholds used for calculating the minimum effort method.
                                        int, default to 8.
    :param m:                           value used for computation of minimum effort thresholds, defaults to 3 as s
                                        uggested in the CUIDADO project.
    :param calculation_type:            method for calculating the attack time, options are 'min_effort' or
                                        'fixed_threshold', default to 'min_effort'.
    :param gradient_calulation_type:    Method for calculating the gradient of the attack, options are 'all' for
                                        calculating the gradient from the estimated start and end points, or 'mean' for
                                        calculating the mean gradient between each threshold step in the minimum effort
                                        method.  Defaults to 'all' and will revert to 'all' if mean is not available.
    :param normalise:                   Normalise the attack segment. bool, default to True.
    :param return_descriptive_data      Default to False, if set to True also returns the thresholds for calculating
                                        the min_effort method.
    :param max_attack_time:             sets the maximum allowable attack time.  Defaults to -1, indicating that there
                                        is no maximum attack time.  This value should be set in seconds.

    :return:                            returns the attack_time, attack_gradient, index of the attack start, and
                                        temporal centroid.
    """
    if normalise:
        # normalise the segments
        normalise_factor = float(max(envelope_samples))
        envelope_samples /= normalise_factor

    if calculate_attack_segment:
        # identify pre-attack segment
        peak_idx = np.argmax(envelope_samples)
        if peak_idx == 0:
            # exit on error
            return 0
        # min_pre_peak_idx = np.argmin(envelope_samples[:peak_idx])
        min_pre_peak_idx = np.where(
            envelope_samples[:peak_idx] == min(envelope_samples[:peak_idx])
        )[-1][-1]

        # redefine the envelope samples as just the min to the peak
        envelope_samples = envelope_samples[min_pre_peak_idx: peak_idx + 1]
    else:
        min_pre_peak_idx = 0

    # calculate the appropriate start and end of the attack using the selected method
    if calculation_type == "min_effort":
        # get threshold time array
        # +2 is to ignore the 0 and 100% levels.
        threshold_step = 1.0 / (thresh_no + 2)
        dyn_range = max(envelope_samples) - min(envelope_samples)
        thresh_level = np.linspace(
            threshold_step, (1 - threshold_step), thresh_no + 1)
        thresh_level = (thresh_level * dyn_range) + min(envelope_samples)

        # predefine an array for when each threshold is crossed
        threshold_idxs = np.zeros(thresh_no + 1)

        # get indexes for when threshold is crossed
        for j in range(len(thresh_level)):
            threshold_hold = np.argmax(envelope_samples >= thresh_level[j])
            # threshold_idxs[j] = threshold_hold + min_pre_peak_idx
            threshold_idxs[j] = threshold_hold

        # calculate effort values (distances between thresholds)
        effort = np.diff(threshold_idxs)

        # get the mean effort value
        effort_mean = np.mean(effort)
        effort_threshold = effort_mean * m

        # find start and stop times foxr the attack
        th_start = np.argmax(effort <= effort_threshold)

        # need to use remaining effort values
        effort_hold = effort[th_start:]
        # this returns a 0 if value not found
        th_end = np.argmax(effort_hold >= effort_threshold)
        if th_end == 0:
            th_end = len(effort_hold) - 1  # make equal to the last value

        # apply correction for holding the values
        th_end = th_end + th_start

        # get the actual start and stop index
        th_start_idx = threshold_idxs[th_start]
        th_end_idx = threshold_idxs[th_end]

        if th_start_idx == th_end_idx:
            th_start_idx = threshold_idxs[0]
            th_end_idx = threshold_idxs[-1]

        if th_start_idx == th_end_idx:
            attack_time = 1.0 / fs
        else:
            attack_time = (th_end_idx - th_start_idx + 1.0) / fs

        if max_attack_time > 0:
            if attack_time > max_attack_time:
                # how many samples is equivalent to the maximum?
                max_attack_time_sample = int(
                    fs * max_attack_time)  # convert to integer
                th_end_idx = th_start_idx + max_attack_time_sample
                attack_time = (th_end_idx - th_start_idx + 1.0) / fs

        start_level = envelope_samples[int(th_start_idx)]
        end_level = envelope_samples[int(th_end_idx)]

        # specify exceptions for a step functions crossing both thresholds
        if start_level == end_level:
            if th_start_idx > 0:
                # if a previous sample is avaiable, take the previous starting sample
                start_level = envelope_samples[int(th_start_idx) - 1]
            else:
                # set start level to zero if onset is at the first sample (indicating a step function at time zero)
                start_level = 0.0

        # is there enough data to calculate the mean
        if gradient_calulation_type == "mean":
            if (end_level - start_level) < 0.2 or (th_end_idx - th_start_idx) < 2:
                # force calculation type to all
                gradient_calulation_type = "all"
                print(
                    "unable to calculate attack gradient with the 'mean' method, reverting to 'all' method."
                )

        if gradient_calulation_type == "mean":
            # calculate the gradient based on the weighted mean of each attack
            threshold_step = dyn_range / (thresh_no + 2)

            gradient_thresh_array = np.arange(
                start_level,
                end_level + (threshold_step * dyn_range),
                (threshold_step * dyn_range),
            )
            cross_threshold_times = np.zeros(len(gradient_thresh_array))
            cross_threshold_values = np.zeros(len(gradient_thresh_array))
            gradient_envelope_segment = envelope_samples[th_start_idx: th_end_idx + 1]

            for i in range(len(cross_threshold_values)):
                hold = np.argmax(gradient_envelope_segment >=
                                 gradient_thresh_array[i])
                cross_threshold_times[i] = hold[0] / float(fs)
                cross_threshold_values[i] = gradient_envelope_segment[hold[0]]

            pente_v = np.diff(cross_threshold_values) / \
                np.diff(cross_threshold_times)

            # calculate weighted average of all gradients with a gausian dsitribution
            m_threshold = 0.5 * \
                (gradient_thresh_array[:-1] + gradient_thresh_array[1:])
            weight_v = np.exp(-((m_threshold - 0.5) ** 2) / (0.5 ** 2))

            attack_gradient = np.sum(pente_v * weight_v) / np.sum(weight_v)

        elif gradient_calulation_type == "all":
            # calculate the attack gradient from th_start_idx to th_end_idx
            attack_gradient = (end_level - start_level) / attack_time

        """
          More stuff to return if we want extra information to be displayed
        """
        thresholds_to_return = [
            calculation_type,
            th_start_idx + min_pre_peak_idx,
            th_end_idx + min_pre_peak_idx,
            threshold_idxs + min_pre_peak_idx,
        ]

    elif calculation_type == "fixed_threshold":
        # set threshold values for fixed threshold method
        fixed_threshold_start = 20
        fixed_threshold_end = 90

        # get dynamic range
        dyn_range = max(envelope_samples) - min(envelope_samples)

        # get thresholds relative to envelope level
        lower_threshold = (fixed_threshold_start * dyn_range * 0.01) + min(
            envelope_samples
        )
        upper_threshold = (fixed_threshold_end * dyn_range * 0.01) + min(
            envelope_samples
        )

        # calculate start index
        th_start_idx = np.argmax(envelope_samples >= lower_threshold)
        # th_start_idx = th_start_idx[0]

        # find the end idx after the start idx
        th_end_idx = np.argmax(
            envelope_samples[th_start_idx:] >= upper_threshold)
        th_end_idx = th_end_idx + th_start_idx

        if th_start_idx == th_end_idx:
            attack_time = 1.0 / fs
        else:
            attack_time = (th_end_idx - th_start_idx + 1.0) / fs

        # compare attack time to maximum permissible attack time
        if max_attack_time > 0:
            if attack_time > max_attack_time:
                # how many samples is equivalent to the maximum?
                max_attack_time_sample = int(
                    fs * max_attack_time)  # convert to integer
                th_end_idx = th_start_idx + max_attack_time_sample
                attack_time = (th_end_idx - th_start_idx + 1.0) / fs

        # calculate the gradient

        # find the level of the first sample used
        start_level = envelope_samples[int(th_start_idx)]
        # find the level of the last sample used
        end_level = envelope_samples[int(th_end_idx)]

        # specify exceptions for a step functions crossing both thresholds
        if start_level == end_level:
            if th_start_idx > 0:
                # if a previous sample is avaiable, take the previous starting sample
                start_level = envelope_samples[int(th_start_idx) - 1]
            else:
                # set start level to zero if onset is at the first sample (indicating a step function at time zero)
                start_level = 0.0

        attack_gradient = (end_level - start_level) / attack_time

        """
          More details to be returned if desired
        """
        thresholds_to_return = [
            calculation_type,
            th_start_idx + min_pre_peak_idx,
            th_end_idx + min_pre_peak_idx,
        ]

    else:
        raise ValueError(
            "calculation_type must be set to either 'fixed_threshold' or 'min_effort'."
        )

    # convert attack time to logarithmic scale
    attack_time = np.log10(attack_time)

    # revert attack gradient metric if envelope has been normalised
    if normalise:
        attack_gradient *= normalise_factor

    """
      Calculate the temporal centroid
    """
    hold_env = envelope_samples[int(th_start_idx): int(th_end_idx) + 1]
    t = np.arange(0, len(hold_env)) / float(fs)
    temp_centroid = np.sum(t * hold_env) / np.sum(hold_env)
    temp_centroid /= float(len(hold_env))

    if return_descriptive_data:
        return (
            attack_time,
            attack_gradient,
            int(th_start_idx + min_pre_peak_idx),
            temp_centroid,
            thresholds_to_return,
        )
    else:
        return (
            attack_time,
            attack_gradient,
            int(th_start_idx + min_pre_peak_idx),
            temp_centroid,
        )


def calculate_onsets(
    audio_samples,
    envelope_samples,
    fs,
    look_back_time=20,
    hysteresis_time=300,
    hysteresis_percent=10,
    onset_in_noise_threshold=10,
    minimum_onset_time_separation=100,
    nperseg=512,
):
    """
      Calculates the onset times using a look backwards recursive function to identify actual note onsets, and weights
       the outputs based on the onset strength to avoid misidentifying onsets.

    Required inputs
    :param audio_samples:                   the audio file in the time domain.
    :param envelope_samples:                the envelope of the audio file, suggested to be calculated with
                                            sample_and_hold_envelope_calculation.
    :param fs:                              samplerate of the audio file.  Function assumes the same sample rate for
                                            both audio_samples and envelop_samples

    Optional inputs
    :param look_back_time:                  time in ms to recursively lookbackwards to identify start of onset,
                                            defaults to 20ms.
    :param hysteresis_time:                 time in ms to look backwards in time for a hysteresis check,
                                            set to 300ms bedefault.
    :param hysteresis_percent:              set the percentage of dynamic range that must be checked when looking
                                            backwards via hysteresis, default to 10%.
    :param onset_in_noise_threshold:        set a threshold of dynamic range for determining if an onset was variation
                                            in noise or an actual onset, default to 10%.
    :param minimum_onset_time_separation:   set the minimum time in ms that two offsets can be separated by.
    :param method:                          set the method for calculating the onsets.  Default to 'librosa', but can
                                            be 'essentia_hfc', or 'essentia_complex'.
    :param nperseg:                         value used in return loop.

    :return:                                thresholded onsets, returns [0] if no onsets are identified.  Note that a
                                            value of [0] is also possible during normal opperation.
    """
    # get onsets with librosa estimation
    onsets = librosa.onset.onset_detect(
        audio_samples, fs, backtrack=True, units="samples"
    )

    # set values for return_loop method
    # 10 ms default look-back time, in samples
    time_thresh = int(look_back_time * 0.001 * fs)
    # hysteresis time, in samples
    hysteresis_samples = int(hysteresis_time * fs * 0.001)
    envelope_dyn_range = max(envelope_samples) - min(envelope_samples)
    hysteresis_thresh = envelope_dyn_range * hysteresis_percent * 0.01

    # only conduct analysis if there are onsets detected
    if np.size(onsets):
        # empty array for storing exact onset idxs
        corrected_onsets = []

        for onset_idx in onsets:
            # if the onset is 1 or 0, it's too close to the start to be corrected (1 is here due to zero padding)
            if onset_idx > 0:
                # actual onset location in samples (librosa uses 512 window size by default)
                onset_loc = np.array(onset_idx).astype("int")

                # only calculate if the onset is NOT at the end of the file, whilst other onsets exist.
                # If the only onset is at the end, calculate anyway.
                if not corrected_onsets:
                    onset_hold = return_loop(
                        onset_loc,
                        envelope_samples,
                        time_thresh,
                        hysteresis_thresh,
                        hysteresis_samples,
                        nperseg=nperseg,
                    )
                    corrected_onsets.append(onset_hold)
                else:
                    if (onset_loc + 511) < len(envelope_samples):
                        onset_hold = return_loop(
                            onset_loc,
                            envelope_samples,
                            time_thresh,
                            hysteresis_thresh,
                            hysteresis_samples,
                            nperseg=nperseg,
                        )
                        corrected_onsets.append(onset_hold)
            else:
                corrected_onsets.append(0)

        # zero is returned from return_loop if no valid onset identified
        # remove zeros (except the first)
        zero_loc = np.where(np.array(corrected_onsets) == 0)[0]
        # ignore if the first value is zero
        if list(zero_loc):
            if zero_loc[0] == 0:
                zero_loc = zero_loc[1:]
        corrected_onsets = np.delete(corrected_onsets, zero_loc)

        # remove duplicates
        hold_onsets = []
        for i in corrected_onsets:
            if i not in hold_onsets:
                hold_onsets.append(i)
        corrected_onsets = hold_onsets

        """
         Remove repeated onsets and compare onset segments against the dynamic range
         to remove erroneous onsets in noise.  If the onset segment (samples between
         adjacent onsets) has a dynamic range less than 10% of total dynamic range,
         remove this onset.
        """
        if len(corrected_onsets) > 1:
            thd_corrected_onsets = []
            last_value = corrected_onsets[-1]
            threshold = onset_in_noise_threshold * envelope_dyn_range * 0.01

            for i in reversed(range(len(corrected_onsets))):
                if corrected_onsets[i] == corrected_onsets[-1]:
                    segment = envelope_samples[corrected_onsets[i]:]
                else:
                    segment = envelope_samples[
                        corrected_onsets[i]: corrected_onsets[i + 1]
                    ]

                # only conduct if the segment if greater than 1 sample long
                if len(segment) > 1:
                    # find attack portion SNR
                    peak_idx = np.argmax(segment)
                    if peak_idx > 0:
                        # get the dynamic range of the attack portion
                        seg_dyn_range = max(segment) - min(segment[:peak_idx])
                        if seg_dyn_range >= threshold:
                            pass
                        else:
                            corrected_onsets = np.delete(corrected_onsets, i)
                    else:
                        corrected_onsets = np.delete(corrected_onsets, i)
                else:
                    corrected_onsets = np.delete(corrected_onsets, i)

        # remove onsets that are too close together, favouring the earlier onset
        if len(corrected_onsets) > 1:
            minimum_onset_time_separation_samples = (
                fs * 0.001 * minimum_onset_time_separation
            )
            time_separation = np.diff(corrected_onsets)
            # while loop for potential multiple itterations
            while (
                len(corrected_onsets) > 1
                and min(time_separation) < minimum_onset_time_separation_samples
            ):
                onsets_to_remove = []
                # some onsets are closer together than the minimum value
                for i in range(len(corrected_onsets) - 1):
                    # are the last two onsets too close?
                    if (
                        abs(corrected_onsets[i + 1] - corrected_onsets[i])
                        < minimum_onset_time_separation_samples
                    ):
                        onsets_to_remove.append(i + 1)

                # remove onsets too close together
                corrected_onsets = np.delete(
                    corrected_onsets, onsets_to_remove)
                time_separation = np.diff(corrected_onsets)

        """
          Correct onsets by comparing to the onset strength.

          If there in an onset strength of 3 or greater between two onsets, then the onset if valid.
          Otherwise, discard the onset.
        """
        thd_corrected_onsets = []

        # get the onset strength
        onset_strength = librosa.onset.onset_strength(audio_samples, fs)

        strength_onset_times = np.array(
            np.array(corrected_onsets) / 512).astype("int")
        strength_onset_times.clip(min=0)

        # corrected_original_onsets = []
        # corrected_strength_onsets = []
        for onset_idx in reversed(range(len(corrected_onsets))):
            current_strength_onset = strength_onset_times[onset_idx]
            if current_strength_onset == strength_onset_times[-1]:
                onset_strength_seg = onset_strength[current_strength_onset:]
            else:
                onset_strength_seg = onset_strength[
                    current_strength_onset: strength_onset_times[onset_idx + 1]
                ]

            if max(onset_strength_seg) < 3:
                strength_onset_times = np.delete(
                    strength_onset_times, onset_idx)
            else:
                thd_corrected_onsets.append(corrected_onsets[onset_idx])

    else:
        return [0]

    thd_corrected_onsets.sort()
    if thd_corrected_onsets:
        return thd_corrected_onsets
    else:
        return [0]


def tf_calculate_onsets(
    audio_samples,
    envelope_samples,
    fs,
    look_back_time=20,
    hysteresis_time=300,
    hysteresis_percent=10,
    onset_in_noise_threshold=10,
    minimum_onset_time_separation=100,
    nperseg=512,
):
    """

    Batch version of calculate_onsets.
      Calculates the onset times using a look backwards recursive function to identify actual note onsets, and weights
       the outputs based on the onset strength to avoid misidentifying onsets.

    Required inputs
    :param audio_samples:                   the audio file in the time domain.
    :param envelope_samples:                the envelope of the audio file, suggested to be calculated with
                                            sample_and_hold_envelope_calculation.
    :param fs:                              samplerate of the audio file.  Function assumes the same sample rate for
                                            both audio_samples and envelop_samples

    Optional inputs
    :param look_back_time:                  time in ms to recursively lookbackwards to identify start of onset,
                                            defaults to 20ms.
    :param hysteresis_time:                 time in ms to look backwards in time for a hysteresis check,
                                            set to 300ms bedefault.
    :param hysteresis_percent:              set the percentage of dynamic range that must be checked when looking
                                            backwards via hysteresis, default to 10%.
    :param onset_in_noise_threshold:        set a threshold of dynamic range for determining if an onset was variation
                                            in noise or an actual onset, default to 10%.
    :param minimum_onset_time_separation:   set the minimum time in ms that two offsets can be separated by.
    :param method:                          set the method for calculating the onsets.  Default to 'librosa', but can
                                            be 'essentia_hfc', or 'essentia_complex'.
    :param nperseg:                         value used in return loop.

    :return:                                thresholded onsets, returns [0] if no onsets are identified.  Note that a
                                            value of [0] is also possible during normal opperation.
    """
    b, _ = audio_samples.shape
    onsets = []
    for i in range(b):
        """
        calculate_onsets(
            audio_samples,
            envelope_samples,
            fs,
            look_back_time=20,
            hysteresis_time=300,
            hysteresis_percent=10,
            onset_in_noise_threshold=10,
            minimum_onset_time_separation=100,
            nperseg=512,
        )"""
        onsets.append(
            tf.numpy_function(
                calculate_onsets,
                [
                    audio_samples[i],
                    envelope_samples[i],
                    fs,
                    look_back_time,
                    hysteresis_time,
                    hysteresis_percent,
                    onset_in_noise_threshold,
                    minimum_onset_time_separation,
                    nperseg,
                ],
                audio_samples.dtype,
                name="calculate_onsets"
            )
        )

    return tf.stack(onsets)


def get_bandwidth_array(
    audio_samples,
    fs,
    nperseg=512,
    overlap_step=32,
    rolloff_thresh=0.01,
    rollon_thresh_percent=0.05,
    log_bandwidth=False,
    return_centroid=False,
    low_bandwidth_method="Percentile",
    normalisation_method="RMS_Time_Window",
):
    """
      Calculate the bandwidth array estimate for an audio signal.

    Required inputs
    :param audio_samples:           array of the audio samples
    :param fs:                      samplerate of the audio samples

    Optional inputs
    :param nperseg:                 numper of samples used for calculating spectrogram
    :param overlap_step:            number of samples overlap for calculating spectrogram
    :param rolloff_thresh:          threshold value for calculating rolloff frequency
    :param rollon_thresh_percent:   percentage threshold for calculating rollon frequency
    :param log_bandwidth:           return the logarithm of the bandwdith, default to False
    :param return_centroid:         return the centroid for each time window
    :param low_bandwidth_method:    method for calculating the low frequency limit of the bandwidth,
                                    default to 'Percentile'
    :param normalisation_method:    method for normlaising the spectrogram, default to 'RMS_Time_Window'

    :return:                        returns the bandwidth array, time array (from spectrogram), and
                                    frequency array (from spectrogram).
    """
    noverlap = nperseg - overlap_step
    # get spectrogram
    f, t, spec = spectrogram(
        audio_samples,
        fs,
        window="boxcar",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude",
    )

    # normalise the spectrogram
    if normalisation_method == "Single_TF_Bin":
        spec /= np.max(spec)
    elif normalisation_method == "RMS_Time_Window":
        spec /= np.max(np.sqrt(np.sum(spec * spec, axis=0)))
    elif normalisation_method == "none":
        pass
    else:
        raise ValueError(
            "Bandwidth normalisation method must be 'Single_TF_Bin' or 'RMS_Time_Window'"
        )

    # get values for thresholding
    level_with_time = np.sum(spec, axis=0)
    max_l = np.max(level_with_time)
    min_l = np.min(level_with_time)
    min_tpower = (0.1 * (max_l - min_l)) + min_l

    # initialise lists for storage
    rollon = []
    rolloff = []
    bandwidth = []
    centroid = []
    centroid_power = []

    # calculate the bandwidth curve
    for time_count in range(len(t)):
        seg = spec[:, time_count]
        tpower = np.sum(seg)
        if tpower > min_tpower:
            if low_bandwidth_method == "Percentile":
                # get the spectral rollon
                rollon_counter = 1
                cumulative_power = np.sum(seg[:rollon_counter])
                rollon_thresh = tpower * rollon_thresh_percent

                while cumulative_power < rollon_thresh:
                    rollon_counter += 1
                    cumulative_power = np.sum(seg[:rollon_counter])
                rollon.append(f[rollon_counter - 1])
            elif low_bandwidth_method == "Cutoff":
                rollon_idx = np.where(seg >= rolloff_thresh)[0]
                if len(rollon_idx):
                    rollon_idx = rollon_idx[0]
                    rollon.append(f[rollon_idx])
            else:
                raise ValueError(
                    "low_bandwidth_method must be 'Percentile' or 'Cutoff'"
                )

            # get the spectral rolloff
            rolloff_idx = np.where(seg >= rolloff_thresh)[0]
            if len(rolloff_idx):
                rolloff_idx = rolloff_idx[-1]
                rolloff.append(f[rolloff_idx])
                if log_bandwidth:
                    bandwidth.append(
                        np.log(f[rolloff_idx] / float(f[rollon_counter - 1]))
                    )
                else:
                    bandwidth.append(f[rolloff_idx] - f[rollon_counter - 1])
            else:
                bandwidth.append(0)

            # get centroid values
            centroid.append(np.sum(seg * f) / np.sum(seg))
            centroid_power.append(tpower)
        else:
            bandwidth.append(0)

    if return_centroid:
        return bandwidth, t, f, np.average(centroid, weights=centroid_power)
    else:
        return bandwidth, t, f


def calculate_bandwidth_gradient(bandwidth_segment, t):
    """
      Calculate the gradient ferom the bandwidth array

    :param bandwidth_segment:   segment of bandwdith for calculation
    :param t:                   time base for calculating

    :return:                    gradient of the bandwidth
    """
    if bandwidth_segment:
        max_idx = np.argmax(bandwidth_segment)
        if max_idx > 0:
            min_idx = np.where(
                np.array(bandwidth_segment[:max_idx])
                == min(bandwidth_segment[:max_idx])
            )[0][-1]

            bandwidth_change = bandwidth_segment[max_idx] - \
                bandwidth_segment[min_idx]
            time_to_change = (max_idx - min_idx) * (t[1] - t[0])

            bandwidth_gradient = bandwidth_change / time_to_change
        else:
            bandwidth_gradient = False
    else:
        bandwidth_gradient = False
    return bandwidth_gradient


def calculate_rms_enveope(
    audio_samples, step_size=256, overlap_step=256, normalise=True
):
    """
      Calculate the RMS envelope of the audio signal.

    :param audio_samples:   numpy array, the audio samples.
    :param step_size:       int, number of samples to get the RMS from.
    :param overlap_step:    int, number of samples to overlap.

    :return:                RMS array
    """
    # initialise lists and counters
    rms_envelope = []
    i = 0
    t_hold = []
    # step through the signal
    while i < len(audio_samples) - step_size:
        rms_envelope.append(
            np.sqrt(
                np.mean(
                    audio_samples[i: i + step_size] *
                    audio_samples[i: i + step_size]
                )
            )
        )
        i += overlap_step

    # use the remainder of the array for a final sample
    t_hold.append(i)
    rms_envelope.append(
        np.sqrt(np.mean(audio_samples[i:] * audio_samples[i:])))
    rms_envelope = np.array(rms_envelope)

    # normalise to peak value
    if normalise:
        rms_envelope = rms_envelope * (1.0 / max(abs(rms_envelope)))

    return rms_envelope


def tf_calculate_rms_enveope(
    audio_samples, step_size=256, overlap_step=256, normalise=True
):
    """
      Calculate the RMS envelope of the audio signal.

    :param audio_samples:   numpy array, the audio samples.
    :param step_size:       int, number of samples to get the RMS from.
    :param overlap_step:    int, number of samples to overlap.

    :return:                RMS array
    """
    rms_envelope = tf.numpy_function(
        calculate_rms_enveope, [audio_samples, step_size, overlap_step, normalise], [tf.float64], name='tf_rms_envelope')
    return rms_envelope


def detect_peaks(array, freq=0, cthr=0.2, unprocessed_array=False, fs=44100):
    """
      Function detects the peaks in array, based from the mirpeaks algorithm.
    :param array:               Array in which to detect peaks
    :param freq:                Scale representing the x axis (sample length as array)
    :param cthr:                Threshold for checking adjacent peaks
    :param unprocessed_array:   Array that in unprocessed (normalised), if False will default to the same as array.
    :param fs:                  Sampe rate of the array

    :return:                     index of peaks, values of peaks, peak value on freq.
    """
    # flatten the array for correct processing
    array = array.flatten()

    if np.isscalar(freq):
        # calculate the frerquency scale - assuming a samplerate if none provided
        freq = np.linspace(0, fs / 2.0, len(array))

    if np.isscalar(unprocessed_array):
        unprocessed_array = array

    # add values to allow peaks at the first and last values
    # to allow peaks at start and end (default of mir)
    array_appended = np.insert(array, [0, len(array)], -2.0)
    # unprocessed array to get peak values
    array_unprocess_appended = np.insert(
        unprocessed_array, [0, len(unprocessed_array)], -2.0
    )
    # append the frequency scale for precise freq calculation
    freq_appended = np.insert(freq, [0, len(freq)], -1.0)

    # get the difference values
    diff_array = np.diff(array_appended)

    # find local maxima
    mx = (
        np.array(
            np.where((array >= cthr) & (
                diff_array[0:-1] > 0) & (diff_array[1:] <= 0))
        )
        + 1
    )

    # initialise arrays for output
    finalmx = []
    peak_value = []
    peak_x = []
    peak_idx = []

    if np.size(mx) > 0:
        # unpack the array if peaks found
        mx = mx[0]

        j = 0  # scans the peaks from beginning to end
        mxj = mx[j]  # the current peak under evaluation
        jj = j + 1
        bufmin = 2.0
        bufmax = array_appended[mxj]

        if mxj > 1:
            oldbufmin = min(array_appended[: mxj - 1])
        else:
            oldbufmin = array_appended[0]

        while jj < len(mx):
            # if adjacent mx values are too close, returns no array
            if mx[jj - 1] + 1 == mx[jj] - 1:
                bufmin = min([bufmin, array_appended[mx[jj - 1]]])
            else:
                bufmin = min(
                    [bufmin, min(array_appended[mx[jj - 1]: mx[jj] - 1])])

            if bufmax - bufmin < cthr:
                # There is no contrastive notch
                if array_appended[mx[jj]] > bufmax:
                    # new peak is significant;y higher than the old peak,
                    # the peak is transfered to the new position
                    j = jj
                    mxj = mx[j]  # the current peak
                    bufmax = array_appended[mxj]
                    oldbufmin = min([oldbufmin, bufmin])
                    bufmin = 2.0
                elif array_appended[mx[jj]] - bufmax <= 0:
                    bufmax = max([bufmax, array_appended[mx[jj]]])
                    oldbufmin = min([oldbufmin, bufmin])

            else:
                # There is a contrastive notch
                if bufmax - oldbufmin < cthr:
                    # But the previous peak candidate is too weak and therefore discarded
                    oldbufmin = min([oldbufmin, bufmin])
                else:
                    # The previous peak candidate is OK and therefore stored
                    finalmx.append(mxj)
                    oldbufmin = bufmin

                bufmax = array_appended[mx[jj]]
                j = jj
                mxj = mx[j]  # The current peak
                bufmin = 2.0

            jj += 1
        if bufmax - oldbufmin >= cthr and (
            bufmax - min(array_appended[mx[j] + 1:]) >= cthr
        ):
            # The last peak candidate is OK and stored
            finalmx.append(mx[j])

        """ Sort the values according to their level """
        finalmx = np.array(finalmx, dtype=np.int64)
        sort_idx = np.argsort(array_appended[finalmx])[::-1]  # descending sort
        finalmx = finalmx[sort_idx]

        # indexes were for the appended array, -1 to return to original array index
        peak_idx = finalmx - 1
        peak_value = array_unprocess_appended[finalmx]
        peak_x = freq_appended[finalmx]

        """ Interpolation for more precise peak location """
        corrected_value = []
        corrected_position = []
        for current_peak_idx in finalmx:
            # if there enough space to do the fitting
            if 1 < current_peak_idx < (len(array_unprocess_appended) - 2):
                y0 = array_unprocess_appended[current_peak_idx]
                ym = array_unprocess_appended[current_peak_idx - 1]
                yp = array_unprocess_appended[current_peak_idx + 1]
                p = (yp - ym) / (2 * (2 * y0 - yp - ym))
                corrected_value.append(y0 - (0.25 * (ym - yp) * p))
                if p >= 0:
                    correct_pos = ((1 - p) * freq_appended[current_peak_idx]) + (
                        p * freq_appended[current_peak_idx + 1]
                    )
                    corrected_position.append(correct_pos)
                elif p < 0:
                    correct_pos = ((1 + p) * freq_appended[current_peak_idx]) - (
                        p * freq_appended[current_peak_idx - 1]
                    )
                    corrected_position.append(correct_pos)
            else:
                corrected_value.append(
                    array_unprocess_appended[current_peak_idx])
                corrected_position.append(freq_appended[current_peak_idx])

        if corrected_position:
            peak_x = corrected_position
            peak_value = corrected_value
        peak_idx = peak_idx.astype(np.int64)
        return peak_idx, np.array(peak_value, dtype=np.float64), np.array(peak_x, np.float64)
    else:
        return np.array([0], dtype=np.int64), np.array(
            [0], dtype=np.float64), np.array([0], np.float64)


def tf_detect_peaks(array, freq=0, cthr=0.2, unprocessed_array=False, fs=44100):
    peak_idx = []
    peak_value = []
    peak_x = []
    b = array.shape[0]
    for k in range(b):
        #peak_idx_, peak_value_, peak_x_
        p = tf.numpy_function(
            detect_peaks, [array[k], freq, cthr, unprocessed_array, fs],)
        peak_idx.append(peak_idx_)
        peak_value.append(peak_value_)
        peak_x.append(peak_x_)

    return peak_idx, peak_value, peak_x


def sigmoid(x, offset=0.2, n=10):
    # return a sigmoidal function for weighting values
    return x ** n / (x ** n + offset)


def tf_sigmoid(x, offset=0.2, n=10):
    NotImplemented


def channel_reduction(audio_samples, phase_correction=False):
    """
      Algorithm for reducing the number of channels in a read-in audio file

    :param audio_samples:       audio samples
    :param phase_correction:    perform phase checking on channels before mono sum

    :return:                    audio samples summed to mono
    """
    # get sum all channels to mono
    num_channels = np.shape(audio_samples)
    if len(num_channels) > 1:
        # check for stereo file
        if num_channels[1] == 2:
            # crudely check for out of phase signals
            if phase_correction:
                r, _ = scipy.stats.pearsonr(
                    audio_samples[:, 0], audio_samples[:, 1])
                if r < -0.5:
                    audio_samples = audio_samples[:, 0]  # [:,1] *= -1.0
                else:
                    audio_samples = np.sum(audio_samples, axis=1)
            else:
                audio_samples = np.sum(audio_samples, axis=1)
        # check for multi-channel file
        elif num_channels[1] > 2:
            # there are multiple layouts for multichannel, I have no way of decoding these with soundfile
            audio_samples = np.sum(audio_samples[:, 0:3], axis=1)

        # TODO Update to include multichannel variants and decode according to: http://www.atsc.org/wp-content/uploads/2015/03/A52-201212-17.pdf
        # elif num_channels[3] > 4:
        # elif num_channels[3] > 5:
        # elif num_channels[3] > 6:

    return audio_samples


def spectral_flux(spectrogram, method="sum"):
    """
      This computes the spectral flux: the difference between sucesive spectrogram time frames

    :param spectrogram:
    :return:
    """
    if method == "sum":
        # sum method
        diff_spec = np.diff(spectrogram, axis=1)  # difference
        sum_flux = np.sqrt(np.sum(diff_spec ** 2, axis=0)) / \
            float(diff_spec.shape[0])

        return sum_flux

    elif method == "multiply":
        # multiplication between adjacent frames
        diff_spec = spectrogram[:, :-1] * spectrogram[:, 1:]
        # variation acorss time
        sum_diff_spec = np.sum(diff_spec ** 2.0, axis=0)
        orig_spec_var = np.sum(spectrogram[:, :-1] ** 2.0, axis=0)
        delayed_spec_var = np.sum(spectrogram[:, 1:] ** 2.0, axis=0)
        # denom = orig_spec_var * delayed_spec_var

        multiply_flux = np.nan_to_num(
            1 - sum_diff_spec / (orig_spec_var * delayed_spec_var)
        )

        return multiply_flux


def log_sum(array):
    """
      This function calculates the log sum of an array

    :param array:
    :return:
    """
    logsum = 10 * np.log10(np.sum(10 ** (array / 10.0)))

    return logsum


def filter_design2(Fc, fs, N):
    """
      Design Butterworth 2nd-order one-third-octave filter.
    """

    f1 = (2.0 ** (-1.0 / 6)) * Fc
    f2 = (2.0 ** (1.0 / 6)) * Fc
    f1 = f1 / (fs / 2.0)
    f2 = f2 / (fs / 2.0)

    # force f2 to be 1.0 for cases where the upper bandwidth from 3rd_octave_downsample produce higher frequencies
    if f2 >= 1.0:
        f2 = 0.9999999999
    b, a = scipy.signal.butter(N, [f1, f2], "bandpass")
    return b, a


def midbands(Fmin, Fmax, fs):
    """
      Divides the frequency range into third octave bands using filters
      Fmin is the minimum third octave band
      Fmax is the maximum third octave band
    """

    # set defaults
    # lowest_band = 25
    # highest_band = 20000
    Nyquist_frequency = fs / 2.0
    # FUpper = (2 ** (1 / 6.0)) * Fmax

    fr = 1000  # reference frequency is 1000Hz
    i = np.arange(-16, 14, 1)
    lab_freq = np.array(
        [
            25,
            31.5,
            40,
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
            6300,
            8000,
            10000,
            12500,
            16000,
            20000,
        ]
    )

    A = np.where(lab_freq == Fmin)[0][0]
    B = np.where(lab_freq == Fmax)[0][0]

    # compare value of B to nyquist
    while lab_freq[B] > Nyquist_frequency:
        B -= 1

    j = i[np.arange(A, B + 1, 1)]  # indices to find exact midband frequencies
    # Exact midband frequencies (Calculated as base two exact)
    ff = (2.0 ** (j / 3.0)) * fr
    F = lab_freq[np.arange(A, B + 1, 1)]
    return ff, F, j


def filter_third_octaves_downsample(x, Pref, fs, Fmin, Fmax, N):
    """
     Filters the audio file into thrid octave bands
     x is the file (Input length must be a multiple of 2^8)
     Pref is the reference level for calculating decibels - does not allow for negative values
     Fmin is the minimum frequency
     Fmax is the maximum frequency (must be at least 2500 Hz)
     Fs is the sampling frequency
     N is the filter order
    """
    # identify midband frequencies
    [ff, F, j] = midbands(Fmin, Fmax, fs)

    # apply filters
    P = np.zeros(len(j))
    # Determines where downsampling will commence (5000 Hz and below)
    k = np.where(j == 7)[0][0]
    m = len(x)

    # For frequencies of 6300 Hz or higher, direct implementation of filters.
    for i in range(len(j) - 1, k, -1):
        B, A = filter_design2(ff[i], fs, N)
        if i == k + 3:  # Upper 1/3-oct. band in last octave.
            Bu = B
            Au = A
        if i == k + 2:  # Center 1/3-oct. band in last octave.
            Bc = B
            Ac = A
        if i == k + 1:  # Lower 1/3-oct. band in last octave.
            Bl = B
            Al = A
        y = scipy.signal.lfilter(B, A, x)
        if np.max(y) > 0:
            # Convert to decibels.
            P[i] = 20 * np.log10(np.sqrt(np.sum(y ** 2.0) / m))
        else:
            P[i] = -1.0 * np.inf

    # 5000 Hz or lower, multirate filter implementation.
    try:
        for i in range(k, 1, -3):  # = k:-3:1;
            # Design anti-aliasing filter (IIR Filter)
            Wn = 0.4
            C, D = scipy.signal.cheby1(2, 0.1, Wn)
            # Filter
            x = scipy.signal.lfilter(C, D, x)
            # Downsample
            idx = np.arange(1, len(x), 2)
            x = x[idx]
            fs = fs / 2.0
            m = len(x)
            # Performs the filtering
            y = scipy.signal.lfilter(Bu, Au, x)
            if np.max(y) > 0:
                P[i] = 20 * np.log10(np.sqrt(np.sum(y ** 2.0) / m))
            else:
                P[i] = -1.0 * np.inf
            y = scipy.signal.lfilter(Bc, Ac, x)
            if np.max(y) > 0:
                P[i - 1] = 20 * np.log10(np.sqrt(np.sum(y ** 2.0) / m))
            else:
                P[i - 1] = -1.0 * np.inf
            y = scipy.signal.lfilter(Bl, Al, x)
            if np.max(y) > 0:
                P[i - 2] = 20 * np.log10(np.sqrt(np.sum(y ** 2.0) / m))
            else:
                P[i - 2] = -1.0 * np.inf
    except:
        P = P[1: len(j)]

    # "calibrate" the readings based from Pref, chosen as 100 in most uses
    P = P + Pref

    # log transformation
    Plog = 10 ** (P / 10.0)
    Ptotal = np.sum(Plog)
    if Ptotal > 0:
        Ptotal = 10 * np.log10(Ptotal)
    else:
        Ptotal = -1.0 * np.inf

    return Ptotal, P, F


def tf_filter_third_octaves_downsample(x, Pref, fs, Fmin, Fmax, N):
    """
     Filters the audio file into thrid octave bands
     x is an audio tensor of format [batch, samples]
     Pref is the reference level for calculating decibels - does not allow for negative values
     Fmin is the minimum frequency
     Fmax is the maximum frequency (must be at least 2500 Hz)
     Fs is the sampling frequency
     N is the filter order
    """
    tf.debugging.assert_rank(x, 2)
    assert fs > 10000, "Enforce fs to be at least 10k"
    # identify midband frequencies
    [ff, F, j] = midbands(Fmin, Fmax, fs)
    # apply filters
    P = np.zeros(len(j))
    # Determines where downsampling will commence (5000 Hz and below)
    k = np.where(j == 7)[0][0]
    m = x.shape[-1]

    # For frequencies of 6300 Hz or higher, direct implementation of filters.
    for i in range(len(j) - 1, k, -1):
        B, A = filter_design2(
            ff[i], fs, N
        )  # describe a band bass butterworth (2nd iorder)
        if i == k + 3:  # Upper 1/3-oct. band in last octave.
            Bu = B
            Au = A
        if i == k + 2:  # Center 1/3-oct. band in last octave.
            Bc = B
            Ac = A
        if i == k + 1:  # Lower 1/3-oct. band in last octave.
            Bl = B
            Al = A
        y = tf_filter(B, A, x)
        if np.max(y) > 0:
            # Convert to decibels.
            P[i] = 20 * np.log10(np.sqrt(np.sum(y ** 2.0) / m))
        else:
            P[i] = -1.0 * np.inf

    # 5000 Hz or lower, multirate filter implementation.
    try:
        for i in range(k, 1, -3):  # = k:-3:1;
            # Design anti-aliasing filter (IIR Filter)
            Wn = 0.4
            C, D = scipy.signal.cheby1(2, 0.1, Wn)
            # Filter
            x = tf_filter(C, D, x)
            # Downsample
            idx = np.arange(1, len(x), 2)
            x = x[idx]
            fs = fs / 2.0
            m = len(x)
            # Performs the filtering
            y = tf_filter(Bu, Au, x)
            if np.max(y) > 0:
                P[i] = 20 * np.log10(np.sqrt(np.sum(y ** 2.0) / m))
            else:
                P[i] = -1.0 * np.inf
            y = tf_filter(Bc, Ac, x)
            if np.max(y) > 0:
                P[i - 1] = 20 * np.log10(np.sqrt(np.sum(y ** 2.0) / m))
            else:
                P[i - 1] = -1.0 * np.inf
            y = scipy.signal.lfilter(Bl, Al, x)
            if np.max(y) > 0:
                P[i - 2] = 20 * np.log10(np.sqrt(np.sum(y ** 2.0) / m))
            else:
                P[i - 2] = -1.0 * np.inf
    except:
        P = P[1: len(j)]

    # "calibrate" the readings based from Pref, chosen as 100 in most uses
    P = P + Pref

    # log transformation
    Plog = 10 ** (P / 10.0)
    Ptotal = np.sum(Plog)
    if Ptotal > 0:
        Ptotal = 10 * np.log10(Ptotal)
    else:
        Ptotal = -1.0 * np.inf

    return Ptotal, P, F


def specific_loudness(x, Pref, fs, Mod):
    """
      Calculates loudness in 3rd octave bands
        based on ISO 532 B / DIN 45631
        Source: BASIC code in J Acoust Soc Jpn(E) 12, 1(1991)
        x = signal
        Pref = refernce value[dB]
        fs = sampling frequency[Hz]
        Mod = 0 for free field
        Mod = 1 for diffuse field

        Returns
        N_entire = entire loudness[sone]
        N_single = partial loudness[sone / Bark]

        Original Matlab code by Claire Churchill Jun. 2004
        Transcoded by Andy Pearce 2018
    """

    # 'Generally used third-octave band filters show a leakage towards neighbouring filters of about -20 dB. This
    # means that a 70dB, 1 - kHz tone produces the following levels at different centre
    # frequencies: 10dB at 500Hz, 30dB at 630Hz, 50dB at 800Hz and 70dB at 1kHz.
    # P211 Psychoacoustics: Facts and Models, E.Zwicker and H.Fastl
    # (A filter order of 4 gives approx this result)

    # set default
    Fmin = 25
    Fmax = 12500
    order = 4
    # filter the audio
    _, P, _ = filter_third_octaves_downsample(x, Pref, fs, Fmin, Fmax, order)

    # set more defaults for perceptual filters

    # Centre frequencies of 1 / 3 Oct bands(FR)

    # Ranges of 1 / 3 Oct bands for correction at low frequencies according to equal loudness contours
    RAP = np.array([45, 55, 65, 71, 80, 90, 100, 120])

    # Reduction of 1/3 Oct Band levels at low frequencies according to equal loudness contours
    # within the eight ranges defined by RAP(DLL)
    DLL = np.array(
        [
            [-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
            [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
            [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
            [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
            [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
            [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
            [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
            [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0],
        ]
    )

    # Critical band level at absolute threshold without taking into account the
    # transmission characteristics of the ear
    # Threshold due to internal noise
    LTQ = np.array([30, 18, 12, 8, 7, 6, 5, 4, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    # Hearing thresholds for the excitation levels (each number corresponds to a critical band 12.5kHz is not included)

    # Attenuation representing transmission between freefield and our hearing system
    A0 = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -
            1.6, -3.2, -5.4, -5.6, -4, -1.5, 2, 5, 12]
    )
    # Attenuation due to transmission in the middle ear
    # Moore et al disagrees with this being flat for low frequencies

    # Level correction to convert from a free field to a diffuse field(last critical band 12.5 kHz is not included)
    DDF = np.array(
        [
            0,
            0,
            0.5,
            0.9,
            1.2,
            1.6,
            2.3,
            2.8,
            3,
            2,
            0,
            -1.4,
            -2,
            -1.9,
            -1,
            0.5,
            3,
            4,
            4.3,
            4,
        ]
    )

    # Correction factor because using third octave band levels(rather than critical bands)
    DCB = np.array(
        [
            -0.25,
            -0.6,
            -0.8,
            -0.8,
            -0.5,
            0,
            0.5,
            1.1,
            1.5,
            1.7,
            1.8,
            1.8,
            1.7,
            1.6,
            1.4,
            1.2,
            0.8,
            0.5,
            0,
            -0.5,
        ]
    )

    # Upper limits of the approximated critical bands
    ZUP = np.array(
        [
            0.9,
            1.8,
            2.8,
            3.5,
            4.4,
            5.4,
            6.6,
            7.9,
            9.2,
            10.6,
            12.3,
            13.8,
            15.2,
            16.7,
            18.1,
            19.3,
            20.6,
            21.8,
            22.7,
            23.6,
            24,
        ]
    )

    # Range of specific loudness for the determination of the steepness of the upper slopes in the specific loudness
    # - critical band rate pattern(used to plot the correct USL curve)
    RNS = np.array(
        [
            21.5,
            18,
            15.1,
            11.5,
            9,
            6.1,
            4.4,
            3.1,
            2.13,
            1.36,
            0.82,
            0.42,
            0.30,
            0.22,
            0.15,
            0.10,
            0.035,
            0,
        ]
    )

    # This is used to design the right hand slope of the loudness
    USL = np.array(
        [
            [13.0, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
            [9.0, 7.5, 6.0, 5.1, 4.5, 4.5, 4.5, 4.5],
            [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
            [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
            [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
            [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
            [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
            [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
            [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
            [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
            [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
            [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
            [0.40, 0.33, 0.26, 0.24, 0.24, 0.22, 0.22, 0.22],
            [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
            [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
            [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
            [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
            [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
        ]
    )

    # apply weighting factors
    Xp = np.zeros(11)
    Ti = np.zeros(11)
    for i in range(11):
        j = 0
        while (P[i] > (RAP[j] - DLL[j, i])) & (j < 7):
            j += 1
        Xp[i] = P[i] + DLL[j, i]
        Ti[i] = 10.0 ** (Xp[i] / 10.0)

    # Intensity values in first three critical bands calculated
    Gi = np.zeros(3)
    # Gi(1) is the first critical band (sum of two octaves(25Hz to 80Hz))
    Gi[0] = np.sum(Ti[0:6])
    # Gi(2) is the second critical band (sum of octave(100Hz to 160Hz))
    Gi[1] = np.sum(Ti[6:9])
    # Gi(3) is the third critical band (sum of two third octave bands(200Hz to 250Hz))
    Gi[2] = np.sum(Ti[9:11])

    if np.max(Gi) > 0.0:
        FNGi = 10 * np.log10(Gi)
    else:
        FNGi = -1.0 * np.inf
    LCB = np.zeros_like(Gi)
    for i in range(3):
        if Gi[i] > 0:
            LCB[i] = FNGi[i]
        else:
            LCB[i] = 0

    # Calculate the main loudness in each critical band
    Le = np.ones(20)
    Lk = np.ones_like(Le)
    Nm = np.ones(21)
    # print("timbarl_loudness P::", P)

    for i in range(20):
        Le[i] = P[i + 8]
        if i <= 2:
            Le[i] = LCB[i]
        Lk[i] = Le[i] - A0[i]
        Nm[i] = 0
        if Mod == 1:
            Le[i] = Le[i] + DDF[i]
        if Le[i] > LTQ[i]:
            Le[i] = Lk[i] - DCB[i]
            S = 0.25
            MP1 = 0.0635 * 10.0 ** (0.025 * LTQ[i])
            MP2 = (1 - S + S * 10 ** (0.1 * (Le[i] - LTQ[i]))) ** 0.25 - 1
            Nm[i] = MP1 * MP2
            if Nm[i] <= 0:
                Nm[i] = 0
    Nm[20] = 0

    KORRY = 0.4 + 0.32 * Nm[0] ** 0.2
    if KORRY > 1:
        KORRY = 1

    Nm[0] = Nm[0] * KORRY

    # Add masking curves to the main loudness in each third octave band
    N = 0
    z1 = 0  # critical band rate starts at 0
    n1 = 0  # loudness level starts at 0
    j = 17
    iz = 0
    z = 0.1
    ns = []

    for i in range(21):
        # Determines where to start on the slope
        ig = i - 1
        if ig > 7:
            ig = 7
        control = 1
        # ZUP is the upper limit of the approximated critical band
        while (z1 < ZUP[i]) | (control == 1):
            # Determines which of the slopes to use
            if n1 < Nm[i]:  # Nm is the main loudness level
                j = 0
                while RNS[j] > Nm[i]:  # the value of j is used below to build a slope
                    # j becomes the index at which Nm(i) is first greater than RNS
                    j += 1

            # The flat portions of the loudness graph
            if n1 <= Nm[i]:
                z2 = ZUP[i]  # z2 becomes the upper limit of the critical band
                n2 = Nm[i]
                N = N + n2 * (z2 - z1)  # Sums the output(N_entire)
                for k in np.arange(z, z2 + 0.01, 0.1):
                    if not ns:
                        ns.append(n2)
                    else:
                        if iz == len(ns):
                            ns.append(n2)
                        elif iz < len(ns):
                            ns[iz] = n2

                    if k < (z2 - 0.05):
                        iz += 1
                z = k  # z becomes the last value of k
                z = round(z * 10) * 0.1

            # The sloped portions of the loudness graph
            if n1 > Nm[i]:
                n2 = RNS[j]
                if n2 < Nm[i]:
                    n2 = Nm[i]
                dz = (n1 - n2) / USL[j, ig]  # USL = slopes
                dz = round(dz * 10) * 0.1
                if dz == 0:
                    dz = 0.1
                z2 = z1 + dz
                if z2 > ZUP[i]:
                    z2 = ZUP[i]
                    dz = z2 - z1
                    n2 = n1 - dz * USL[j, ig]  # USL = slopes
                N = N + dz * (n1 + n2) / 2.0  # Sums the output(N_entire)
                for k in np.arange(z, z2 + 0.01, 0.1):
                    if not ns:
                        ns.append(n1 - (k - z1) * USL[j, ig])
                    else:
                        if iz == len(ns):
                            ns.append(n1 - (k - z1) * USL[j, ig])
                        elif iz < len(ns):
                            ns[iz] = n1 - (k - z1) * USL[j, ig]
                    if k < (z2 - 0.05):
                        iz += 1
                z = k
                z = round(z * 10) * 0.1
            if n2 == RNS[j]:
                j += 1
            if j > 17:
                j = 17
            n1 = n2
            z1 = z2
            z1 = round(z1 * 10) * 0.1
            control += 1

    if N < 0:
        N = 0

    if N <= 16:
        N = np.floor(N * 1000 + 0.5) / 1000.0
    else:
        N = np.floor(N * 100 + 0.05) / 100.0

    LN = 40.0 * (N + 0.0005) ** 0.35

    if LN < 3:
        LN = 3

    if N >= 1:
        LN = 10 * np.log10(N) / np.log10(2) + 40

    N_single = np.zeros(240)
    for i in range(240):
        N_single[i] = ns[i]

    N_entire = N
    return N_entire, N_single


@tf.function
def tf_specific_loudness2(x, Pref, fs, Mod):
    b = x.shape[0]
    out = []
    print(x.shape)
    for i in range(b):
        out.append(tf.numpy_function(specific_loudness,
                                     [x[i], Pref, fs, Mod], tf.float64, name="tf_specific_loudness"))

    return out


@tf.function
def tf_specific_loudness(x, Pref, fs, Mod):

    # 'Generally used third-octave band filters show a leakage towards neighbouring filters of about -20 dB. This
    # means that a 70dB, 1 - kHz tone produces the following levels at different centre
    # frequencies: 10dB at 500Hz, 30dB at 630Hz, 50dB at 800Hz and 70dB at 1kHz.
    # P211 Psychoacoustics: Facts and Models, E.Zwicker and H.Fastl
    # (A filter order of 4 gives approx this result)

    # set default
    Fmin = 25
    Fmax = 12500
    order = 4
    # filter the audio
    _, P, _ = tf.numpy_function(
        filter_third_octaves_downsample,
        [x, Pref, fs, Fmin, Fmax, order],
        [tf.float64, tf.float64, tf.float64], name="filter_octave_downsample"
    )
    P = tf.cast(P, x.dtype)
    # NOTE ::assuming P is [samples, ...]
    # set more defaults for perceptual filters

    # Ranges of 1 / 3 Oct bands for correction at low frequencies according to equal loudness contours
    RAP = K.constant([45, 55, 65, 71, 80, 90, 100, 120])

    # Reduction of 1/3 Oct Band levels at low frequencies according to equal loudness contours
    # within the eight ranges defined by RAP(DLL)
    DLL = K.constant(
        [
            [-32, -24, -16, -10, -5, 0, -7, -3, 0, -2, 0],
            [-29, -22, -15, -10, -4, 0, -7, -2, 0, -2, 0],
            [-27, -19, -14, -9, -4, 0, -6, -2, 0, -2, 0],
            [-25, -17, -12, -9, -3, 0, -5, -2, 0, -2, 0],
            [-23, -16, -11, -7, -3, 0, -4, -1, 0, -1, 0],
            [-20, -14, -10, -6, -3, 0, -4, -1, 0, -1, 0],
            [-18, -12, -9, -6, -2, 0, -3, -1, 0, -1, 0],
            [-15, -10, -8, -4, -2, 0, -3, -1, 0, -1, 0],
        ]
    )

    # Critical band level at absolute threshold without taking into account the
    # transmission characteristics of the ear
    # Threshold due to internal noise
    LTQ = K.constant([30, 18, 12, 8, 7, 6, 5, 4, 3,
                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    # Hearing thresholds for the excitation levels (each number corresponds to a critical band 12.5kHz is not included)

    # Attenuation representing transmission between freefield and our hearing system
    A0 = K.constant(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.5, -
            1.6, -3.2, -5.4, -5.6, -4, -1.5, 2, 5, 12]
    )
    # Attenuation due to transmission in the middle ear
    # Moore et al disagrees with this being flat for low frequencies

    # Level correction to convert from a free field to a diffuse field(last critical band 12.5 kHz is not included)
    DDF = K.constant(
        [
            0,
            0,
            0.5,
            0.9,
            1.2,
            1.6,
            2.3,
            2.8,
            3,
            2,
            0,
            -1.4,
            -2,
            -1.9,
            -1,
            0.5,
            3,
            4,
            4.3,
            4,
        ]
    )

    # Correction factor because using third octave band levels(rather than critical bands)
    DCB = K.constant(
        [
            -0.25,
            -0.6,
            -0.8,
            -0.8,
            -0.5,
            0,
            0.5,
            1.1,
            1.5,
            1.7,
            1.8,
            1.8,
            1.7,
            1.6,
            1.4,
            1.2,
            0.8,
            0.5,
            0,
            -0.5,
        ]
    )

    # Upper limits of the approximated critical bands
    ZUP = K.constant(
        [
            0.9,
            1.8,
            2.8,
            3.5,
            4.4,
            5.4,
            6.6,
            7.9,
            9.2,
            10.6,
            12.3,
            13.8,
            15.2,
            16.7,
            18.1,
            19.3,
            20.6,
            21.8,
            22.7,
            23.6,
            24,
        ]
    )

    # Range of specific loudness for the determination of the steepness of the upper slopes in the specific loudness
    # - critical band rate pattern(used to plot the correct USL curve)
    RNS = K.constant(
        [
            21.5,
            18,
            15.1,
            11.5,
            9,
            6.1,
            4.4,
            3.1,
            2.13,
            1.36,
            0.82,
            0.42,
            0.30,
            0.22,
            0.15,
            0.10,
            0.035,
            0,
        ]
    )

    # This is used to design the right hand slope of the loudness
    USL = K.constant(
        [
            [13.0, 8.2, 6.3, 5.5, 5.5, 5.5, 5.5, 5.5],
            [9.0, 7.5, 6.0, 5.1, 4.5, 4.5, 4.5, 4.5],
            [7.8, 6.7, 5.6, 4.9, 4.4, 3.9, 3.9, 3.9],
            [6.2, 5.4, 4.6, 4.0, 3.5, 3.2, 3.2, 3.2],
            [4.5, 3.8, 3.6, 3.2, 2.9, 2.7, 2.7, 2.7],
            [3.7, 3.0, 2.8, 2.35, 2.2, 2.2, 2.2, 2.2],
            [2.9, 2.3, 2.1, 1.9, 1.8, 1.7, 1.7, 1.7],
            [2.4, 1.7, 1.5, 1.35, 1.3, 1.3, 1.3, 1.3],
            [1.95, 1.45, 1.3, 1.15, 1.1, 1.1, 1.1, 1.1],
            [1.5, 1.2, 0.94, 0.86, 0.82, 0.82, 0.82, 0.82],
            [0.72, 0.67, 0.64, 0.63, 0.62, 0.62, 0.62, 0.62],
            [0.59, 0.53, 0.51, 0.50, 0.42, 0.42, 0.42, 0.42],
            [0.40, 0.33, 0.26, 0.24, 0.24, 0.22, 0.22, 0.22],
            [0.27, 0.21, 0.20, 0.18, 0.17, 0.17, 0.17, 0.17],
            [0.16, 0.15, 0.14, 0.12, 0.11, 0.11, 0.11, 0.11],
            [0.12, 0.11, 0.10, 0.08, 0.08, 0.08, 0.08, 0.08],
            [0.09, 0.08, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05],
            [0.06, 0.05, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
        ]
    )

    # apply weighting factors
    Xp = []
    Ti = []
    for i in range(11):
        j = 0
        while (P[i] > (RAP[j] - DLL[j, i])) & (j < 7):
            j += 1
        Xp.append(P[i] + DLL[j, i])
        Ti.append(10.0 ** (Xp[i] / 10.0))

    Xp = tf.stack(Xp)
    Ti = tf.stack(Ti)

    # Intensity values in first three critical bands calculated
    Gi = []
    # Gi(1) is the first critical band (sum of two octaves(25Hz to 80Hz))
    Gi.append(K.sum(Ti[0:6]))
    # Gi(2) is the second critical band (sum of octave(100Hz to 160Hz))
    Gi.append(K.sum(Ti[6:9]))
    # Gi(3) is the third critical band (sum of two third octave bands(200Hz to 250Hz))
    Gi.append(K.sum(Ti[9:11]))
    Gi = tf.stack(Gi)

    if K.max(Gi) > 0.0:
        FNGi = 10 * log10(Gi)
    else:
        FNGi = Gi.dtype.min
    LCB = []
    for i in range(3):
        LCB.append(FNGi[i])
    tf.stack(LCB)

    # Calculate the main loudness in each critical band
    Le = []
    Lk = []
    Nm = []
    for i in range(20):
        # Le[i] = P[i + 8]
        Le.append(P[i + 8])
        if i <= 2:
            Le[i] = LCB[i]
        Lk.append(Le[i] - A0[i])
        Nm.append(0.0)
        if Mod == 1:
            Le[i] = Le[i] + DDF[i]
        if Le[i] > LTQ[i]:
            Le[i] = Lk[i] - DCB[i]
            S = 0.25
            MP1 = 0.0635 * 10.0 ** (0.025 * LTQ[i])
            MP2 = (1 - S + S * 10 ** (0.1 * (Le[i] - LTQ[i]))) ** 0.25 - 1
            Nm[i] = MP1 * MP2
            if Nm[i] <= 0:
                Nm[i] = tf.cast(0, Nm[i].dtype)
        if i == 0:
            KORRY = 0.4 + 0.32 * Nm[0] ** 0.2
            if KORRY > 1:
                KORRY = 1.0
            Nm[i] = Nm[i] * KORRY

    Nm.append(0)
    Le = tf.stack(Le)
    Lk = tf.stack(Lk)
    Nm = tf.stack(Nm)

    KORRY = 0.4 + 0.32 * Nm[0] ** 0.2
    if KORRY > 1:
        KORRY = 1.0

    # Nm[0] = Nm[0] * KORRY
    # Add masking curves to the main loudness in each third octave band
    N = 0
    z1 = 0  # critical band rate starts at 0
    n1 = 0  # loudness level starts at 0
    j = 17
    iz = 0
    z = 0.1
    ns = []
    for ii in range(21):
        # Determines where to start on the slope
        ig = ii - 1
        if ig > 7:
            ig = 7
        control = 1
        # ZUP is the upper limit of the approximated critical band
        zup = ZUP[ii]
        nm = Nm[ii]
        # while (z1 < ZUP[ii]) | (control == 1):
        while (z1 < zup) | (control == 1):
            # Determines which of the slopes to use
            if n1 < Nm[ii]:  # Nm is the main loudness level
                j = 0
                while RNS[j] > Nm[ii]:  # the value of j is used below to build a slope
                    # j becomes the index at which Nm(i) is first greater than RNS
                    j += 1

            # The flat portions of the loudness graph
            if n1 <= Nm[ii]:
                z2 = ZUP[ii]  # z2 becomes the upper limit of the critical band
                n2 = Nm[ii]
                N = N + n2 * (z2 - z1)  # Sums the output(N_entire)
                for k in np.arange(z, z2 + 0.01, 0.1):
                    if not ns:
                        ns.append(n2)
                    else:
                        if iz == len(ns):
                            ns.append(n2)
                        elif iz < len(ns):
                            ns[iz] = n2

                    if k < (z2 - 0.05):
                        iz += 1
                z = k  # z becomes the last value of k
                z = round(z * 10) * 0.1

            # The sloped portions of the loudness graph
            if n1 > Nm[ii]:
                n2 = RNS[j]
                if n2 < Nm[ii]:
                    n2 = Nm[ii]
                dz = (n1 - n2) / USL[j, ig]  # USL = slopes
                dz = tf.math.round(dz * 10) * 0.1
                if dz == 0:
                    dz = 0.1
                z2 = z1 + dz
                if z2 > ZUP[ii]:
                    z2 = ZUP[ii]
                    dz = z2 - z1
                    n2 = n1 - dz * USL[j, ig]  # USL = slopes
                N = N + dz * (n1 + n2) / 2.0  # Sums the output(N_entire)
                for k in np.arange(z, z2 + 0.01, 0.1):
                    if not ns:
                        ns.append(n1 - (k - z1) * USL[j, ig])
                    else:
                        if iz == len(ns):
                            ns.append(n1 - (k - z1) * USL[j, ig])
                        elif iz < len(ns):
                            ns[iz] = n1 - (k - z1) * USL[j, ig]
                    if k < (z2 - 0.05):
                        iz += 1
                z = k
                z = round(z * 10) * 0.1
            if n2 == RNS[j]:
                j += 1
            if j > 17:
                j = 17
            n1 = n2
            z1 = z2
            z1 = tf.math.round(z1 * 10) * 0.1
            control += 1

    if N < 0:
        N = 0

    if N <= 16:
        N = np.floor(N * 1000 + 0.5) / 1000.0
    else:
        N = np.floor(N * 100 + 0.05) / 100.0

    LN = 40.0 * (N + 0.0005) ** 0.35

    if LN < 3:
        LN = 3

    if N >= 1:
        LN = 10 * np.log10(N) / np.log10(2) + 40

    N_single = np.zeros(240)
    for i in range(240):
        N_single[i] = ns[i]

    N_entire = N
    return N_entire, N_single


def output_clip(score, min_score=0, max_score=100):
    """
      Limits the output of the score between min_score and max_score

    :param score:
    :param min_score:
    :param max_score:
    :return:
    """
    """
    if score < min_score:
        return 0.0
    elif score > max_score:
        return 100.0
    else:
        return score
    """
    if tf.is_tensor(score):
        return K.clip(score, min_score, max_score)
    else:
        return np.clip(score, min_score, max_score)


def fast_hilbert(array, use_matlab_hilbert=False):
    """
      Calculates the hilbert transform of the array by segmenting signal first to speed up calculation.
    :param array:
    :return:
    """
    step_size = 32768
    overlap = 2
    overlap_size = int(step_size / (2 * overlap))
    # how many steps, rounded to nearest int
    # step_no = int((len(array) / (step_size - overlap)) + 0.5)
    step_start = 0
    hold_hilbert = np.array([])
    while (step_start + step_size) < len(array):
        hold_array = array[step_start: step_start + step_size]
        if use_matlab_hilbert:
            this_hilbert = np.abs(matlab_hilbert(hold_array))
        else:
            this_hilbert = np.abs(scipy.signal.hilbert(hold_array))

        if step_start == 0:
            # try to concatonate the results
            hold_hilbert = np.concatenate(
                (hold_hilbert, this_hilbert[: 3 * overlap_size])
            )
        else:
            hold_hilbert = np.concatenate(
                (hold_hilbert, this_hilbert[overlap_size: 3 * overlap_size])
            )

        # increment the step
        step_start += int(step_size / overlap)

    # do the last step
    hold_array = array[step_start:]
    this_hilbert = np.abs(scipy.signal.hilbert(hold_array))

    # try to concatonate the results
    hold_hilbert = np.concatenate((hold_hilbert, this_hilbert[overlap_size:]))
    return hold_hilbert


def fast_hilbert_spectrum(array, use_matlab_hilbert=False):
    """
      Calculates the hilbert transform of the array by segmenting signal first to speed up calculation.
    :param array:
    :return:
    """
    step_size = 32768
    overlap = 2
    overlap_size = int(step_size / (2 * overlap))
    step_start = 0
    hold_HILBERT = []
    if (step_start + step_size) < len(array):
        while (step_start + step_size) < len(array):
            hold_array = array[step_start: step_start + step_size]
            if use_matlab_hilbert:
                this_hilbert = np.abs(matlab_hilbert(hold_array))
            else:
                this_hilbert = np.abs(scipy.signal.hilbert(hold_array))

            HILBERT = np.abs(np.fft.fft(np.abs(this_hilbert)))
            HILBERT = HILBERT[0: int(len(HILBERT) / 2.0)]  # take the real part
            hold_HILBERT.append(HILBERT)

            step_start += int(step_size / overlap)

        # hilbert_spectrum = np.sum(hold_HILBERT, axis=0)
        hilbert_spectrum = np.mean(hold_HILBERT, axis=0)

    else:
        # how much to pad by
        array = np.pad(
            array, (0, step_size - len(array)), "constant", constant_values=0.0
        )

        if use_matlab_hilbert:
            this_hilbert = np.abs(matlab_hilbert(array))
        else:
            this_hilbert = np.abs(scipy.signal.hilbert(array))

        HILBERT = np.abs(np.fft.fft(np.abs(this_hilbert)))
        HILBERT = HILBERT[0: int(len(HILBERT) / 2.0)]  # take the real part

        hilbert_spectrum = HILBERT

    return hilbert_spectrum


def matlab_hilbert(signal):
    """
      Define a method for calculating the hilbert transform of a 1D array using the method from Matlab

    :param signal:
    :return:
    """
    # get the fft
    n = len(signal)
    x = np.fft.fft(signal)
    h = np.zeros(n)

    if (n > 0) and (~isodd(n)):
        # even and nonempty
        h[0] = 1
        h[int(n / 2)] = 1
        h[1: int(n / 2)] = 2
    elif n > 0:
        # odd and nonempty
        h[0] = 1
        h[1: int((n + 1) / 2.0)] = 2

    # this is the hilbert bit
    x = np.fft.ifft(x * h)

    return x


def isodd(num):
    return num & 0x1


def window_audio(audio_samples, window_length=4096):
    """
      Segment the audio samples into a numpy array the correct size and shape, so that each row is a new window of audio
    :param audio_samples:
    :param window_length:
    :param overlap:
    :return:
    """
    remainder = np.mod(
        len(audio_samples), window_length
    )  # how many samples are left after division
    if remainder > 0:
        # zero pad audio samples
        audio_samples = np.pad(
            audio_samples,
            (0, int(window_length - remainder)),
            "constant",
            constant_values=0.0,
        )
    windowed_samples = np.reshape(
        audio_samples, (int(len(audio_samples) / window_length),
                        int(window_length))
    )

    return windowed_samples


def tf_window_audio(audio_samples, window_length=4096):
    """
      Segment the audio samples into a numpy array the correct size and shape, so that each row is a new window of audio
    :param audio_samples: Tensor, should be of format BN
    :param window_length:
    :param overlap:
    :return:
    """
    return tf.signal.frame(audio_samples, window_length, window_length, pad_end=True)


def normal_dist(array, theta=1.0, mean=0.0):
    y = (1.0 / (theta * np.sqrt(2.0 * np.pi))) * np.exp(
        (-1.0 * ((array - mean) ** 2.0)) / 2.0 * (theta ** 2.0)
    )
    return y


def weighted_bark_level(audio_samples, fs, low_bark_band=0, upper_bark_band=240):
    # window the audio
    windowed_samples = window_audio(audio_samples)

    # need to define a function for the roughness stimuli, emphasising the 20 - 40 region (of the bark scale)
    mean_bark_band = (low_bark_band + upper_bark_band) / 2.0
    array = np.arange(low_bark_band, upper_bark_band)
    x = normal_dist(array, theta=0.01, mean=mean_bark_band)
    x -= np.min(x)
    x /= np.max(x)

    weight_array = np.zeros(240)
    weight_array[low_bark_band:upper_bark_band] = x

    windowed_loud_spec = []
    windowed_rms = []
    weighted_vals = []

    for i in range(windowed_samples.shape[0]):
        samples = windowed_samples[i, :]
        _, N_single = specific_loudness(samples, Pref=100.0, fs=fs, Mod=0)

        # append the loudness spec
        windowed_loud_spec.append(N_single)
        windowed_rms.append(np.sqrt(np.mean(samples * samples)))
        weighted_vals.append(np.sum(weight_array * N_single))

    mean_weight = np.mean(weighted_vals)
    weighted_weight = np.average(weighted_vals, weights=windowed_rms)

    return mean_weight, weighted_weight


def tf_weighted_bark_level(audio_samples, fs, low_bark_band=0, upper_bark_band=240):
    b = audio_samples.shape[0]
    mean_weight = []
    weighted_weight = []
    for k in range(b):
        t1, t2 = tf.numpy_function(weighted_bark_level, [
                                   audio_samples, fs, low_bark_band, upper_bark_band], tf.float64, name="weighted_bark_level")
        mean_weight.append(t1)
        weighted_weight.append(t2)

    return mean_weight, weighted_weight


"""
  Loudnorm function to be included in future update
"""


def loud_norm(audio, fs=44100, target_loudness=-24.0):
    """
      Takes in audio data and returns the same audio loudness normalised
    :param audio:
    :param fs:
    :param target_loudness:
    :return:
    """
    meter = pyln.Meter(fs)

    # minimum length of file is 0.4 seconds
    if len(audio) < (fs * 0.4):
        # how much longer does the file need to be?
        samples_needed = int(fs * 0.4) - len(audio)

        # zero pad signal
        len_check_audio = np.pad(
            audio, (0, samples_needed), "constant", constant_values=0.0
        )
    else:
        len_check_audio = audio

    # assess the current loudness
    current_loudness = meter.integrated_loudness(len_check_audio)
    normalised_audio = pyln.normalize.loudness(
        audio, current_loudness, target_loudness)

    # check for clipping and reduce level
    if np.max(np.abs(normalised_audio)) > 1.0:
        normalised_audio /= np.max(np.abs(normalised_audio))

    return normalised_audio


def file_read(
    fname,
    fs=0,
    phase_correction=False,
    mono_sum=True,
    loudnorm=True,
    resample_low_fs=True,
):
    """
      Read in audio file, but check if it's already an array
      Return samples if already an array.
    :param fname:
    :return:
    """
    if isinstance(fname, six.string_types):
        # use pysoundfile to read audio
        audio_samples, fs = sf.read(fname, always_2d=False)

    elif hasattr(fname, "shape"):
        if fs == 0:
            raise ValueError(
                "If giving function an array, 'fs' must be specified")
        audio_samples = fname

    else:
        raise TypeError(
            "Input type of 'fname' must be string, or have a shape attribute (e.g. a numpy array)"
        )

    # check audio file contains data
    if audio_samples.size == 0:
        raise ValueError("Input audio file does not contain data")

    # reduce to mono
    if mono_sum:
        audio_samples = channel_reduction(audio_samples, phase_correction)

    # check data has values
    if np.max(np.abs(audio_samples)) == 0.0:
        raise ValueError("Input file is silence, cannot be analysed.")

    # loudness normalise
    if loudnorm:
        audio_samples = loud_norm(audio_samples, fs, target_loudness=-24.0)

    if resample_low_fs:
        # check if upsampling required and perform to avoid errors
        audio_samples, fs = check_upsampling(audio_samples, fs)

    return audio_samples, fs


def check_upsampling(audio_samples, fs, lowest_fs=44100):
    """
      Check if upsampling needfs to be applied, then perform it if necessary

    :param audio_samples:
    :param fs:
    :return:
    """
    if fs < lowest_fs:
        # upsample file to avoid errors when calculating specific loudness
        audio_samples = librosa.core.resample(audio_samples, fs, lowest_fs)
        fs = lowest_fs

    return audio_samples, fs


def tf_log10(x, dtype=None):
    """
    x : tensor
    dtype : enforce the type for the output. Default to None, giving the type of x to the output.
    """
    base = tf.cast(tf.math.log(10.0), x.dtype if dtype is None else dtype)
    x = tf.cast(x, dtype) if dtype else x
    return tf.math.log(x) / base


def high_shelf(fc, Q, gain, fs):
    # high shelf formula
    # TODO :: refactor
    A = np.sqrt(gain)
    wc = 2 * np.pi * fc / fs
    wS = np.sin(wc)
    wC = np.cos(wc)
    beta = np.sqrt(A) / Q

    a0 = (A + 1.0) - ((A - 1.0) * wC) + (beta * wS)

    b = np.zeros(3)
    a = np.zeros(3)
    b[0] = A * ((A + 1.0) + ((A - 1.0) * wC) + (beta * wS)) / a0
    b[1] = -2.0 * A * ((A - 1.0) + ((A + 1.0) * wC)) / a0
    b[2] = A * ((A + 1.0) + ((A - 1.0) * wC) - (beta * wS)) / a0

    a[0] = 1
    a[1] = 2.0 * ((A - 1.0) - ((A + 1.0) * wC)) / a0
    a[2] = ((A + 1.0) - ((A - 1.0) * wC) - (beta * wS)) / a0
    return b, a
