from __future__ import division
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow.keras.backend as K
from . import timbral_util


def tf_tri(N, M=None, k=0, dtype=float, *, like=None):
    """
    An array with ones at and below the given diagonal and zeros elsewhere.
    Parameters
    ----------
    N : int
        Number of rows in the array.
    M : int, optional
        Number of columns in the array.
        By default, `M` is taken equal to `N`.
    k : int, optional
        The sub-diagonal at and below which the array is filled.
        `k` = 0 is the main diagonal, while `k` < 0 is below it,
        and `k` > 0 is above.  The default is 0.
    dtype : dtype, optional
        Data type of the returned array.  The default is float.
    ${ARRAY_FUNCTION_LIKE}
        .. versionadded:: 1.20.0
    Returns
    -------
    tri : ndarray of shape (N, M)
        Array with its lower triangle filled with ones and zero elsewhere;
        in other words ``T[i,j] == 1`` for ``j <= i + k``, 0 otherwise.
    Examples
    --------
    >>> np.tri(3, 5, 2, dtype=int)
    array([[1, 1, 1, 0, 0],
           [1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1]])
    >>> np.tri(3, 5, -1)
    array([[0.,  0.,  0.,  0.,  0.],
           [1.,  0.,  0.,  0.,  0.],
           [1.,  1.,  0.,  0.,  0.]])
    """

    if M is None:
        M = N
    m = tf.math.greater_equal(tf.range(0, N),
                              tf.range(-k, M-k))
    m = m * tf.transpose(m)

    # print(m)

    # Avoid making a copy if the requested type is already bool

    return m


def tf_tril(m, k=0):
    """
    Lower triangle of an array.
    Return a copy of an array with elements above the `k`-th diagonal zeroed.
    Parameters
    ----------
    m : array_like, shape (M, N)
        Input array.
    k : int, optional
        Diagonal above which to zero elements.  `k = 0` (the default) is the
        main diagonal, `k < 0` is below it and `k > 0` is above.
    Returns
    -------
    tril : ndarray, shape (M, N)
        Lower triangle of `m`, of same shape and data-type as `m`.
    See Also
    --------
    triu : same thing, only for the upper triangle
    Examples
    --------
    >>> np.tril([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], -1)
    array([[ 0,  0,  0],
           [ 4,  0,  0],
           [ 7,  8,  0],
           [10, 11, 12]])
    """
    mask = tf_tri(*m.shape[-2:], k=k, dtype=bool)

    return tf.where(mask, m, tf.zeros(1, m.dtype))


def plomp(f1, f2):
    """
      Plomp's algorithm for estimating roughness.

    :param f1:  float, frequency of first frequency of the pair
    :param f2:  float, frequency of second frequency of the pair
    :return:
    """
    b1 = 3.51
    b2 = 5.75
    xstar = 0.24
    s1 = 0.0207
    s2 = 18.96
    s = np.tril(xstar / ((s1 * np.minimum(f1, f2)) + s2))
    pd = np.exp(-b1 * s * np.abs(f2 - f1)) - np.exp(-b2 * s * np.abs(f2 - f1))
    return pd


def tf_plomp(f1, f2):
    """
      Plomp's algorithm for estimating roughness.

    :param f1:  float, frequency of first frequency of the pair
    :param f2:  float, frequency of second frequency of the pair
    :return:
    """
    b1 = 3.51
    b2 = 5.75
    xstar = 0.24
    s1 = 0.0207
    s2 = 18.96
    s = tf_tril(xstar / ((s1 * K.minimum(f1, f2)) + s2))
    pd = K.exp(-b1 * s * K.abs(f2 - f1)) - K.exp(-b2 * s * K.abs(f2 - f1))
    return pd


def timbral_roughness(
    fname,
    dev_output=False,
    phase_correction=False,
    clip_output=False,
    fs=0,
    peak_picking_threshold=0.01,
    take_first=None

):
    """
     This function is an implementation of the Vassilakis [2007] model of roughness.
     The peak picking algorithm implemented is based on the MIR toolbox's implementation.

     This version of timbral_roughness contains self loudness normalising methods and can accept arrays as an input
     instead of a string filename.

     Version 0.4


     Vassilakis, P. 'SRA: A Aeb-based researh tool for spectral and roughness analysis of sound signals', Proceedings
     of the 4th Sound and Music Computing Conference, Lefkada, Greece, July, 2007.

     Required parameter
      :param fname:                 string, Audio filename to be analysed, including full file path and extension.

     Optional parameters
      :param dev_output:            bool, when False return the roughness, when True return all extracted features
                                    (current none).
      :param phase_correction:      bool, if the inter-channel phase should be estimated when performing a mono sum.
                                    Defaults to False.

      :return:                      Roughness of the audio signal.

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
    """
      Pad audio
    """
    # pad audio
    audio_samples = np.lib.pad(
        audio_samples, (512, 0), "constant", constant_values=(0.0, 0.0)
    )
    """
      Reshape audio into time windows of 50ms.
    """
    # reshape audio
    audio_len = len(audio_samples)
    time_step = 0.05
    step_samples = int(fs * time_step)
    nfft = step_samples
    window = np.hamming(nfft + 2)
    window = window[1:-1]
    olap = nfft / 2
    num_frames = int((audio_len) / (step_samples - olap))
    next_pow_2 = np.log(step_samples) / np.log(2)
    next_pow_2 = 2 ** int(next_pow_2 + 1)

    reshaped_audio = np.zeros([next_pow_2, num_frames])

    i = 0
    start_idx = int((i * (nfft / 2.0)))

    # check if audio is too short to be reshaped
    if audio_len > step_samples:
        # get all the audio
        while start_idx + step_samples <= audio_len:
            audio_frame = audio_samples[start_idx: start_idx + step_samples]

            # apply window
            audio_frame = audio_frame * window

            # append zeros
            reshaped_audio[:step_samples, i] = audio_frame

            # increase the step
            i += 1
            start_idx = int((i * (nfft / 2.0)))
    else:
        # reshaped audio is just padded audio samples
        reshaped_audio[:audio_len, i] = audio_samples
    spec = np.fft.fft(reshaped_audio, axis=0)
    spec_len = int(next_pow_2 / 2) + 1
    spec = spec[:spec_len, :]
    spec = np.absolute(spec)
    freq = fs / 2 * np.linspace(0, 1, spec_len)

    # normalise spectrogram based from peak TF bin
    norm_spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

    """ Peak picking algorithm """
    cthr = peak_picking_threshold  # threshold for peak picking

    _, no_segments = np.shape(spec)

    allpeakpos = []
    allpeaklevel = []
    allpeaktime = []
    # print("acm no segment", no_segments)
    for i in range(0, no_segments):
        d = norm_spec[:, i]
        d_un = spec[:, i]

        # find peak candidates
        peak_pos, peak_level, peak_x = timbral_util.detect_peaks(
            d, cthr=cthr, unprocessed_array=d_un, freq=freq
        )
        allpeakpos.append(peak_pos)
        allpeaklevel.append(peak_level)
        allpeaktime.append(peak_x)

    """ Calculate the Vasillakis Roughness """
    allroughness = []
    # for each frame
    for frame in range(len(allpeaklevel)):
        frame_freq = allpeaktime[frame]
        frame_level = allpeaklevel[frame]

        if len(frame_freq) > 1:
            # Looks very much like a repeat to me....
            f2 = np.kron(np.ones([len(frame_freq), 1]), frame_freq)
            f1 = f2.T
            v2 = np.kron(np.ones([len(frame_level), 1]), frame_level)
            v1 = v2.T

            X = v1 * v2
            Y = (2 * v2) / (v1 + v2)
            Z = plomp(f1, f2)
            rough = (X ** 0.1) * (0.5 * (Y ** 3.11)) * Z

            allroughness.append(np.sum(rough))
        else:
            allroughness.append(0)

    mean_roughness = np.mean(allroughness)

    if dev_output:
        return [mean_roughness]
    else:
        """
          Perform linear regression
        """
        print("acm_mean_roughness", mean_roughness)
        # cap roughness for low end
        if mean_roughness < 0.01:
            return 0
        else:
            roughness = np.log10(mean_roughness) * \
                13.98779569 + 48.97606571545886
            if clip_output:
                roughness = timbral_util.output_clip(roughness)

            return roughness


# @tf.function
def tf_timbral_roughness(
    audio_tensor,
    dev_output=False,
    phase_correction=False,
    clip_output=False,
    fs=0,
    peak_picking_threshold=0.01,
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
    """
      Read input
    """
    assert fs, "fs should be provided to tf_timbral_roughness"
    # tf.debugging.assert_rank(audio_tensor, 2)
    audio_samples, fs = audio_tensor, fs
    audio_samples = audio_samples[:, :, 0]
    fs = float(fs)
    """
      Pad audio
    """
    # pad audio
    audio_samples = tf.pad(
        audio_samples, [[0, 0, ], [512, 0]], "constant", constant_values=0)

    """
      Reshape audio into time windows of 50ms.
    """
    # reshape audio
    audio_len = len(audio_samples[1])
    time_step = 0.05
    step_samples = int(fs * time_step)
    nfft = step_samples
    window = np.hamming(nfft + 2)
    window = window[1:-1]
    olap = nfft // 2
    num_frames = int((audio_len) / (step_samples - olap))
    next_pow_2 = K.log(float(step_samples)) / K.log(float(2))
    next_pow_2 = 2 ** tf.cast(next_pow_2 + 1, tf.int32)

    # reshaped_audio = tf.frame(audio_samples, step_samples, step_samples)

    i = 0
    start_idx = int((i * (nfft / 2.0)))
    print(audio_len)
    """
    # check if audio is too short to be reshaped
    if audio_len > step_samples:
        # get all the audio
        while start_idx + step_samples <= audio_len:
            audio_frame = audio_samples[0, start_idx: start_idx + step_samples]
            # apply window
            audio_frame = audio_frame * window

            # append zeros
            reshaped_audio[:step_samples, i] = audio_frame

            # increase the step
            i += 1
            start_idx = int((i * (nfft / 2.0)))
    else:
        # reshaped audio is just padded audio samples
        reshaped_audio[:audio_len, i] = audio_samples

    spec = np.fft.fft(reshaped_audio, axis=0)
    spec_len = int(next_pow_2 / 2)+1
    spec = spec[:spec_len, :]
    spec_2 = np.absolute(spec)
    """
    freq, _, spec = timbral_util.compat_spectrogram(
        audio_samples,
        fs,
        "hamming",
        nfft,
        olap,
        nfft,
        False,
        True,
        "none",
    )

    # freq = fs / 2 * np.linspace(0, 1, spec_len)
    # TODO :: compat_spectrogram should work here.
    # normalise spectrogram based from peak TF bin
    b_norm = 1/(K.max(spec, axis=[1, 2], keepdims=True) -
                K.min(spec, axis=[1, 2], keepdims=True))
    norm_spec = b_norm * (spec - K.min(spec, axis=[1, 2], keepdims=True))

    """ Peak picking algorithm """
    cthr = peak_picking_threshold  # threshold for peak picking

    b, no_segments, _ = np.shape(spec)

    k = 0
    all_roughness_array = []
    # TODO arrays to apend values
    for k in range(b):
        allpeakpos = []
        allpeaklevel = []
        allpeaktime = []
        for i in range(no_segments):
            # causes loss of gradient
            d = norm_spec[k, i, :]
            d_un = spec[k, i, :]
            # find peak candidates
            #
            pp = tf.numpy_function(timbral_util.detect_peaks,
                                   [d, freq, cthr, d_un,  fs], [
                                       tf.int64, tf.float64, tf.float64]
                                   )
            if len(pp) == 3:
                peak_pos, peak_level, peak_x = pp[0], pp[1], pp[2]
                allpeakpos.append(peak_pos)
                allpeaklevel.append(peak_level)
                allpeaktime.append(peak_x)
        """ Calculate the Vasillakis Roughness """
        allroughness = []
        # for each frame
        for frame in range(len(allpeaklevel)):
            frame_freq = allpeaktime[frame]
            frame_level = allpeaklevel[frame]
            # print("frame_freq, frame_level :: ",                  frame_freq.shape, frame_level.shape)
            if len(frame_freq) > 1:
                # f2 = np.kron(np.ones([len(frame_freq), 1]), frame_freq)
                f2 = tf.tile(tf.expand_dims(
                    frame_freq, axis=0), [len(frame_freq), 1])
                # print("f2 ::", np.shape(f2))
                f1 = tf.transpose(f2)
                # v2 = np.kron(np.ones([len(frame_level), 1]), frame_level)
                # v1 = v2.T
                v2 = tf.tile(tf.expand_dims(
                    frame_level, axis=0), [len(frame_level), 1])
                # print("v2 ::", np.shape(v2))
                v1 = tf.transpose(v2)
                X = v1 * v2
                Y = (2 * v2) / (v1 + v2)
                Z = tf.numpy_function(
                    plomp, [f1, f2], [tf.float64], name="plomp")
                rough = tf.cast((X ** 0.1) * (0.5 * (Y ** 3.11))
                                * Z, audio_samples.dtype)
            else:
                rough = tf.cast(0, audio_samples.dtype)
            allroughness.append(K.sum(rough))

        all_roughness_array.append(allroughness)

    mean_roughness = K.mean(tf.stack(all_roughness_array), axis=-1)

    if dev_output:
        return [mean_roughness]
    else:
        """
          Perform linear regression
        """
        # cap roughness for low end
        print("tf_mean_roughness", mean_roughness)
        roughness = timbral_util.tf_log10(mean_roughness) * \
            13.98779569 + 48.97606571545886
        if clip_output:
            roughness = timbral_util.output_clip(roughness)

        return roughness
