from __future__ import division
import numpy as np
import librosa
import soundfile as sf
import six
from scipy.signal import spectrogram
from . import timbral_util
import tensorflow as tf
import tensorflow.keras.backend as K


def timbral_hardness(fname, fs=0, dev_output=False, phase_correction=False, clip_output=False, max_attack_time=0.1,
                     bandwidth_thresh_db=-75, take_first=None):
    """
     This function calculates the apparent hardness of an audio file.
     This version of timbral_hardness contains self loudness normalising methods and can accept arrays as an input
     instead of a string filename.

     Version 0.4

     Required parameter
      :param fname:                 string or numpy array
                                    string, audio filename to be analysed, including full file path and extension.
                                    numpy array, array of audio samples, requires fs to be set to the sample rate.

     Optional parameters
      :param fs:                    int/float, when fname is a numpy array, this is a required to be the sample rate.
                                    Defaults to 0.
      :param phase_correction:      bool, perform phase checking before summing to mono.  Defaults to False.
      :param dev_output:            bool, when False return the depth, when True return all extracted
                                    features.  Default to False.
      :param clip_output:           bool, force the output to be between 0 and 100.
      :param max_attack_time:       float, set the maximum attack time, in seconds.  Defaults to 0.1.
      :param bandwidth_thresh_db:   float, set the threshold for calculating the bandwidth, Defaults to -75dB.


      :return:                      float, Apparent hardness of audio file, float (dev_output = False/default).
                                    With dev_output set to True returns the weighted mean bandwidth,
                                    mean attack time, harmonic-percussive ratio, and unitless attack centroid.

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

    '''
      Read input
    '''
    audio_samples, fs = timbral_util.file_read(
        fname, fs, phase_correction=phase_correction)
    if take_first:
        audio_samples = audio_samples[:take_first]
    '''
      Calculate the midband level
    '''
    # get the level in the midband
    midband_level, weighed_midband_level = timbral_util.weighted_bark_level(audio_samples, fs, low_bark_band=70,
                                                                            upper_bark_band=140)
    log_weighted_midband_level = np.log10(weighed_midband_level)

    '''
      Calculate the harmonic-percussive ratio pre zero-padding the signal
    '''
    HP_ratio = timbral_util.get_percussive_audio(
        audio_samples, return_ratio=True)
    log_HP_ratio = np.log10(HP_ratio)

    '''
     Zeropad the signal
    '''
    # zero pad the signal
    nperseg = 4096  # default value for spectrogram analysis
    audio_samples = np.lib.pad(
        audio_samples, (nperseg+1, 0), 'constant', constant_values=(0.0, 0.0))

    '''
      Calculate the envelope and onsets
    '''
    # calculate the envelope of the signal
    envelope = timbral_util.sample_and_hold_envelope_calculation(
        audio_samples, fs, decay_time=0.1)
    envelope_time = np.arange(len(envelope)) / fs

    # calculate the onsets
    original_onsets = timbral_util.calculate_onsets(
        audio_samples, envelope, fs, nperseg=nperseg)
    onset_strength = librosa.onset.onset_strength(audio_samples, fs)
    # If onsets don't exist, set it to time zero
    if not original_onsets:
        original_onsets = [0]
    # set to start of file in the case where there is only one onset
    if len(original_onsets) == 1:
        original_onsets = [0]

    onsets = np.array(original_onsets) - nperseg
    onsets[onsets < 0] = 0

    '''
      Calculate the spectrogram so that the bandwidth can be created
    '''
    bandwidth_step_size = 128
    # calculate threshold in linear from dB
    mag = timbral_util.db2mag(bandwidth_thresh_db)
    bandwidth, t, f = timbral_util.get_bandwidth_array(audio_samples, fs, nperseg=nperseg,
                                                       overlap_step=bandwidth_step_size, rolloff_thresh=mag,
                                                       normalisation_method='none')
    # bandwidth sample rate
    # fs due to spectrogram step size
    bandwidth_fs = fs / float(bandwidth_step_size)

    '''
      Set all parameters for holding data per onset
    '''
    all_bandwidth_max = []
    all_attack_time = []
    all_max_strength = []
    all_max_strength_bandwidth = []
    all_attack_centroid = []

    '''
      Get bandwidth onset times and max bandwidth
    '''
    bandwidth_onset = np.array(
        onsets / float(bandwidth_step_size)).astype('int')  # overlap_step=128

    '''
      Iterate through onsets and calculate metrics for each
    '''
    for onset_count in range(len(bandwidth_onset)):
        '''
          Calculate the bandwidth max for the attack portion of the onset
        '''
        # get the section of the bandwidth array between onsets
        onset = bandwidth_onset[onset_count]
        if onset == bandwidth_onset[-1]:
            bandwidth_seg = np.array(bandwidth[onset:])
        else:
            next_onset = bandwidth_onset[onset_count + 1]
            bandwidth_seg = np.array(bandwidth[onset:next_onset])

        if max(bandwidth_seg) > 0:
            # making a copy of the bandqwidth segment to avoid array changes
            hold_bandwidth_seg = list(bandwidth_seg)

            # calculate onset of the attack in the bandwidth array
            if max(bandwidth_seg) > 0:
                bandwidth_attack = timbral_util.calculate_attack_time(bandwidth_seg, bandwidth_fs,
                                                                      calculation_type='fixed_threshold',
                                                                      max_attack_time=max_attack_time)
            else:
                bandwidth_attack = []

            # calculate the badiwdth max for the attack portion
            if bandwidth_attack:
                start_idx = bandwidth_attack[2]
                if max_attack_time > 0:
                    max_attack_time_samples = int(
                        max_attack_time * bandwidth_fs)
                    if len(hold_bandwidth_seg[start_idx:]) > start_idx+max_attack_time_samples:
                        all_bandwidth_max.append(
                            max(hold_bandwidth_seg[start_idx:start_idx+max_attack_time_samples]))
                    else:
                        all_bandwidth_max.append(
                            max(hold_bandwidth_seg[start_idx:]))
                else:
                    all_bandwidth_max.append(
                        max(hold_bandwidth_seg[start_idx:]))
        else:
            # set as blank so bandwith
            bandwidth_attack = []

        '''
          Calculate the attack time
        '''
        onset = original_onsets[onset_count]
        if onset == original_onsets[-1]:
            attack_seg = np.array(envelope[onset:])
            # 512 is librosa default window size
            strength_seg = np.array(onset_strength[int(onset/512):])
            audio_seg = np.array(audio_samples[onset:])
        else:
            attack_seg = np.array(
                envelope[onset:original_onsets[onset_count + 1]])
            strength_seg = np.array(
                onset_strength[int(onset/512):int(original_onsets[onset_count+1]/512)])
            audio_seg = np.array(
                audio_samples[onset:original_onsets[onset_count + 1]])

        attack_time = timbral_util.calculate_attack_time(
            attack_seg, fs, max_attack_time=max_attack_time)
        all_attack_time.append(attack_time[0])

        '''
          Get the attack strength for weighting the bandwidth max
        '''
        all_max_strength.append(max(strength_seg))
        if bandwidth_attack:
            all_max_strength_bandwidth.append(max(strength_seg))

        '''
          Get the spectral centroid of the attack (125ms after attack start)
        '''
        # identify the start of the attack
        th_start_idx = attack_time[2]
        # define how long the attack time can be
        # number of samples for attack time integration
        centroid_int_samples = int(0.125 * fs)

        # start of attack section from attack time calculation
        if th_start_idx + centroid_int_samples >= len(audio_seg):
            audio_seg = audio_seg[th_start_idx:]
        else:
            audio_seg = audio_seg[th_start_idx:th_start_idx +
                                  centroid_int_samples]

        # check that there's a suitable legnth of samples to get attack centroid
        # minimum length arbitrarily set to 512 samples
        if len(audio_seg) > 512:
            # get all spectral features for this attack section
            spectral_features_hold = timbral_util.get_spectral_features(
                audio_seg, fs)

            # store unitless attack centroid if exists
            if spectral_features_hold:
                all_attack_centroid.append(spectral_features_hold[0])

    '''
      Calculate mean and weighted average values for features
    '''
    # attack time
    mean_attack_time = np.mean(all_attack_time)

    # get the weighted mean of bandwidth max and limit lower value
    if len(all_bandwidth_max):
        mean_weighted_bandwidth_max = np.average(
            all_bandwidth_max, weights=all_max_strength_bandwidth)
        # check for zero values so the log bandwidth max can be taken
        if mean_weighted_bandwidth_max <= 512.0:
            mean_weighted_bandwidth_max = fs / 512.0  # minimum value
    else:
        mean_weighted_bandwidth_max = fs / 512.0  # minimum value

    # take the logarithm
    log_weighted_bandwidth_max = np.log10(mean_weighted_bandwidth_max)

    # get the mean of the onset strenths
    mean_max_strength = np.mean(all_max_strength)
    log_mean_max_strength = np.log10(mean_max_strength)

    if all_attack_centroid:
        mean_attack_centroid = np.mean(all_attack_centroid)
    else:
        mean_attack_centroid = 200.0

    # limit the lower limit of the attack centroid to allow for log to be taken
    if mean_attack_centroid <= 200:
        mean_attack_centroid = 200.0
    log_attack_centroid = np.log10(mean_attack_centroid)

    '''
      Either return the raw features, or calculaste the linear regression.
    '''
    if dev_output:
        return log_weighted_bandwidth_max, log_attack_centroid, log_weighted_midband_level, log_HP_ratio, log_mean_max_strength, mean_attack_time
    else:
        '''
         Apply regression model
        '''
        all_metrics = np.ones(7)
        all_metrics[0] = log_weighted_bandwidth_max
        all_metrics[1] = log_attack_centroid
        all_metrics[2] = log_weighted_midband_level
        all_metrics[3] = log_HP_ratio
        all_metrics[4] = log_mean_max_strength
        all_metrics[5] = mean_attack_time

        # coefficients = np.array([13.5330599736, 18.1519030059, 13.1679266873, 5.03134507433, 5.22582123237, -3.71046018962, -89.8935449357])

        # recalculated values when using loudnorm
        coefficients = np.array([12.079781720638145, 18.52100377170042, 14.139883645260355, 5.567690321917516,
                                 3.9346817690405635, -4.326890461087848, -85.60352209068202])

        hardness = np.sum(all_metrics * coefficients)

        # clip output between 0 and 100
        if clip_output:
            hardness = timbral_util.output_clip(hardness)

        return hardness


def tf_timbral_hardness(audio_tensor, fs, dev_output=False, phase_correction=False, clip_output=False, max_attack_time=0.1,
                        bandwidth_thresh_db=-75):
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

    '''
      Read input
    '''
    audio_samples, fs = audio_tensor, fs
    assert tf.rank(audio_samples) == 2 or tf.rank(
        audio_samples) == 3, "audio_tensor should be of rank 2 or 3. Got {}".format(tf.rank(audio_samples))

    if tf.rank(audio_samples) == 3:
        audio_samples_ = audio_samples[:, :, 0]
    else:
        audio_samples_ = audio_samples
    b = audio_samples_.shape[0]
    fs = float(fs)
    '''
      Calculate the midband level
    '''
    log_weighted_bandwidth_max_array = []
    log_attack_centroid_array = []
    log_weighted_midband_level_array = []
    log_HP_ratio_array = []
    log_mean_max_strength_array = []
    mean_attack_time_array = []
    for audio_samples in audio_samples_:
        # get the level in the midband
        _, weighed_midband_level = tf.numpy_function(timbral_util.weighted_bark_level, [audio_samples, fs, 70,
                                                                                        140], [tf.float32, tf.float32])
        log_weighted_midband_level = timbral_util.tf_log10(
            weighed_midband_level, dtype=audio_samples.dtype)

        '''
        Calculate the harmonic-percussive ratio pre zero-padding the signal
        '''
        HP_ratio = tf.numpy_function(timbral_util.get_percussive_audio, [
                                     audio_samples, True], [tf.float64])
        log_HP_ratio = timbral_util.tf_log10(HP_ratio)

        '''
        Zeropad the signal
        '''
        # zero pad the signal
        nperseg = 4096  # default value for spectrogram analysis
        audio_samples = np.lib.pad(
            audio_samples, (nperseg+1, 0), 'constant', constant_values=(0.0, 0.0))

        '''
        Calculate the envelope and onsets
        '''
        # calculate the envelope of the signal
        envelope = tf.numpy_function(timbral_util.sample_and_hold_envelope_calculation, [
            audio_samples, fs, 0.1, 0.01], [tf.float64])
        envelope_time = tf.range(len(envelope), dtype=audio_samples.dtype) / fs
        # Simplification : un seul onset
        """
        # calculate the onsets
        original_onsets = timbral_util.tf_calculate_onsets(
            audio_samples, envelope, fs, nperseg=nperseg)
        onset_strength = librosa.onset.onset_strength(audio_samples, fs)
        # If onsets don't exist, set it to time zero
        if not original_onsets:
            original_onsets = [0]
        # set to start of file in the case where there is only one onset
        if len(original_onsets) == 1:
            original_onsets = [0]

        onsets = np.array(original_onsets) - nperseg
        onsets[onsets < 0] = 0
        """
        '''
        Calculate the spectrogram so that the bandwidth can be created
        '''
        # Simplification :: one onset
        onsets = np.array([0])
        original_onsets = onsets
        onset_strength = tf.numpy_function(librosa.onset.onset_strength, [
                                           audio_samples, fs], [tf.float32])

        bandwidth_step_size = 128
        # calculate threshold in linear from dB
        mag = timbral_util.db2mag(bandwidth_thresh_db)
        bandwidth, _, _ = timbral_util.get_bandwidth_array(audio_samples, fs, nperseg=nperseg,
                                                           overlap_step=bandwidth_step_size, rolloff_thresh=mag,
                                                           normalisation_method='none')
        # bandwidth sample rate
        # fs due to spectrogram step size
        bandwidth_fs = fs / float(bandwidth_step_size)

        '''
        Set all parameters for holding data per onset
        '''
        all_bandwidth_max = []
        all_attack_time = []
        all_max_strength = []
        all_max_strength_bandwidth = []
        all_attack_centroid = []

        '''
        Get bandwidth onset times and max bandwidth
        '''
        bandwidth_onset = np.array(
            onsets / float(bandwidth_step_size)).astype('int')  # overlap_step=128
        bandwidth_onset = [0]
        '''
        Iterate through onsets and calculate metrics for each
        '''
        for onset_count in range(len(bandwidth_onset)):
            '''
            Calculate the bandwidth max for the attack portion of the onset
            '''
            # get the section of the bandwidth array between onsets
            onset = bandwidth_onset[onset_count]
            if onset == bandwidth_onset[-1]:
                bandwidth_seg = np.array(bandwidth[onset:])
            else:
                next_onset = bandwidth_onset[onset_count + 1]
                bandwidth_seg = np.array(bandwidth[onset:next_onset])

            if max(bandwidth_seg) > 0:
                # making a copy of the bandqwidth segment to avoid array changes
                hold_bandwidth_seg = list(bandwidth_seg)

                # calculate onset of the attack in the bandwidth array
                if max(bandwidth_seg) > 0:
                    bandwidth_attack = timbral_util.calculate_attack_time(bandwidth_seg, bandwidth_fs,
                                                                          calculation_type='fixed_threshold',
                                                                          max_attack_time=max_attack_time)
                else:
                    bandwidth_attack = []

                # calculate the badiwdth max for the attack portion
                if bandwidth_attack:
                    start_idx = bandwidth_attack[2]
                    if max_attack_time > 0:
                        max_attack_time_samples = int(
                            max_attack_time * bandwidth_fs)
                        if len(hold_bandwidth_seg[start_idx:]) > start_idx+max_attack_time_samples:
                            all_bandwidth_max.append(
                                K.max(hold_bandwidth_seg[start_idx:start_idx+max_attack_time_samples]))
                        else:
                            all_bandwidth_max.append(
                                K.max(hold_bandwidth_seg[start_idx:]))
                    else:
                        all_bandwidth_max.append(
                            K.max(hold_bandwidth_seg[start_idx:]))
            else:
                # set as blank so bandwith
                bandwidth_attack = []

            '''
            Calculate the attack time
            '''
            onset = original_onsets[onset_count]
            if onset == original_onsets[-1]:
                attack_seg = np.array(envelope[onset:])
                # 512 is librosa default window size
                strength_seg = np.array(onset_strength[int(onset/512):])
                audio_seg = np.array(audio_samples[onset:])
            else:
                attack_seg = np.array(
                    envelope[onset:original_onsets[onset_count + 1]])
                strength_seg = np.array(
                    onset_strength[int(onset/512):int(original_onsets[onset_count+1]/512)])
                audio_seg = np.array(
                    audio_samples[onset:original_onsets[onset_count + 1]])

            attack_time = timbral_util.calculate_attack_time(
                attack_seg, fs, max_attack_time=max_attack_time)
            all_attack_time.append(attack_time[0])

            '''
            Get the attack strength for weighting the bandwidth max
            '''
            all_max_strength.append(max(strength_seg))
            if bandwidth_attack:
                all_max_strength_bandwidth.append(max(strength_seg))

            '''
            Get the spectral centroid of the attack (125ms after attack start)
            '''
            # identify the start of the attack
            th_start_idx = attack_time[2]
            # define how long the attack time can be
            # number of samples for attack time integration
            centroid_int_samples = int(0.125 * fs)

            # start of attack section from attack time calculation
            if th_start_idx + centroid_int_samples >= len(audio_seg):
                audio_seg = audio_seg[th_start_idx:]
            else:
                audio_seg = audio_seg[th_start_idx:th_start_idx +
                                      centroid_int_samples]

            # check that there's a suitable legnth of samples to get attack centroid
            # minimum length arbitrarily set to 512 samples
            if len(audio_seg) > 512:
                # get all spectral features for this attack section
                spectral_features_hold = timbral_util.get_spectral_features(
                    audio_seg, fs)

                # store unitless attack centroid if exists
                if spectral_features_hold:
                    all_attack_centroid.append(spectral_features_hold[0])

        '''
        Calculate mean and weighted average values for features
        '''
        # attack time
        mean_attack_time = tf.cast(
            K.mean(tf.stack(all_attack_time)), audio_samples.dtype)

        all_max_strength_bandwidth = tf.stack(all_max_strength_bandwidth)
        all_bandwidth_max = tf.cast(
            tf.stack(all_bandwidth_max), all_max_strength_bandwidth.dtype)
        # get the weighted mean of bandwidth max and limit lower value
        if len(all_bandwidth_max):
            print(all_bandwidth_max, all_max_strength_bandwidth)
            mean_weighted_bandwidth_max = timbral_util.tf_average(
                all_bandwidth_max, all_max_strength_bandwidth)
            # check for zero values so the log bandwidth max can be taken
            if mean_weighted_bandwidth_max <= 512.0:
                mean_weighted_bandwidth_max = fs / 512.0  # minimum value
        else:
            mean_weighted_bandwidth_max = fs / 512.0  # minimum value

        # take the logarithm
        log_weighted_bandwidth_max = timbral_util.tf_log10(
            mean_weighted_bandwidth_max)

        # get the mean of the onset strenths
        mean_max_strength = np.mean(all_max_strength)
        log_mean_max_strength = np.log10(mean_max_strength)
        all_attack_centroid = tf.stack(all_attack_centroid)
        if all_attack_centroid:
            mean_attack_centroid = K.mean(all_attack_centroid)
        else:
            mean_attack_centroid = 200.0

        # limit the lower limit of the attack centroid to allow for log to be taken
        if mean_attack_centroid <= 200:
            mean_attack_centroid = 200.0
        log_attack_centroid = timbral_util.tf_log10(
            mean_attack_centroid, tf.float32)
        # Appending the results
        log_weighted_bandwidth_max_array.append(log_weighted_bandwidth_max)
        log_attack_centroid_array.append(log_attack_centroid)
        log_weighted_midband_level_array.append(log_weighted_midband_level)
        log_HP_ratio_array.append(log_HP_ratio)
        log_mean_max_strength_array.append(log_mean_max_strength)
        mean_attack_time_array.append(mean_attack_time)

    '''
      Either return the raw features, or calculaste the linear regression.
    '''
    if dev_output:
        return log_weighted_bandwidth_max, log_attack_centroid, log_weighted_midband_level, log_HP_ratio, log_mean_max_strength, mean_attack_time
    else:
        '''
         Apply regression model
        '''
        ####
        log_weighted_bandwidth_max = tf.stack(
            log_weighted_bandwidth_max_array)  # 1 instead of 2
        log_attack_centroid = tf.stack(
            log_attack_centroid_array)  # is float64 no  more
        log_weighted_midband_level = tf.stack(
            log_weighted_midband_level_array)  # is float64 no more
        log_HP_ratio = tf.stack(log_HP_ratio_array)
        log_mean_max_strength = tf.stack(log_mean_max_strength_array)
        mean_attack_time = tf.stack(mean_attack_time_array)  # is float64
        ####
        print("metrics :: ", log_weighted_bandwidth_max, log_attack_centroid, log_weighted_midband_level,
              log_HP_ratio, log_mean_max_strength, mean_attack_time)
        # coefficients = np.array([13.5330599736, 18.1519030059, 13.1679266873, 5.03134507433, 5.22582123237, -3.71046018962, -89.8935449357])

        # recalculated values when using loudnorm
        coefficients = np.array([12.079781720638145, 18.52100377170042, 14.139883645260355, 5.567690321917516,
                                 3.9346817690405635, -4.326890461087848, -85.60352209068202])

        hardness = 12.079781720638145 * log_weighted_bandwidth_max
        """+ 18.52100377170042*log_attack_centroid + 14.139883645260355*log_weighted_midband_level + \
            5.567690321917516*log_HP_ratio + 3.9346817690405635 * log_mean_max_strength - \
            4.326890461087848*mean_attack_time - 85.60352209068202
        """
        # clip output between 0 and 100
        if clip_output:
            hardness = timbral_util.output_clip(hardness)

        return hardness
