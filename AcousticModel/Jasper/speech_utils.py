# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math
import os

import h5py
import numpy as np
import resampy as rs
import scipy.io.wavfile as wave
BACKENDS = []
try:
  import python_speech_features as psf
  BACKENDS.append('psf')
except ImportError:
  pass
try:
  import librosa
  BACKENDS.append('librosa')
except ImportError:
  pass

WINDOWS_FNS = {"hanning": np.hanning, "hamming": np.hamming, "none": None}

def get_speech_features_from_file(filename, params, return_wav=False):
  """Function to get a numpy array of features, from an audio file.
      if params['cache_features']==True, try load preprocessed data from
      disk, or store after preprocesseng.
      else, perform preprocessing on-the-fly.

  Args:
    filename (string): WAVE filename.
    params (dict): the following parameters
      num_features (int): number of speech features in frequency domain.
      features_type (string): 'mfcc' or 'spectrogram'.
      window_size (float): size of analysis window in milli-seconds.
      window_stride (float): stride of analysis window in milli-seconds.
      augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`augment_audio_signal` for specification and example.
      window (str): window function to apply
      dither (float): weight of Gaussian noise to apply to input signal for
          dithering/preventing quantization noise
      num_fft (int): size of fft window to use if features require fft,
          defaults to smallest power of 2 larger than window size
      norm_per_feature (bool): if True, the output features will be normalized
          (whitened) individually. if False, a global mean/std over all features
          will be used for normalization
  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
  """
  # sample_freq, signal = wave.read(filename)
  signal, sample_freq = librosa.load(filename, sr=16000)
  features, duration, wav = get_speech_features(signal, sample_freq, params)

  if return_wav:
    return features, duration, wav
  return features, duration


def normalize_signal(signal, gain=None):
  """
  Normalize float32 signal to [-1, 1] range
  """
  if gain is None:
    gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
  return signal * gain


def augment_audio_signal(signal_float, sample_freq, augmentation):
  """Function that performs audio signal augmentation.

  Args:
    signal_float (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    augmentation (dict, optional): None or dictionary of augmentation parameters.
        If not None, has to have 'speed_perturbation_ratio',
        'noise_level_min', or 'noise_level_max' fields, e.g.::
          augmentation={
            'speed_perturbation_ratio': 0.2,
            'noise_level_min': -90,
            'noise_level_max': -46,
          }
        'speed_perturbation_ratio' can either be a list of possible speed
        perturbation factors or a float. If float, a random value from 
        U[1-speed_perturbation_ratio, 1+speed_perturbation_ratio].
  Returns:
    np.array: np.array with augmented audio signal.
  """
  if 'speed_perturbation_ratio' in augmentation:
    stretch_amount = -1
    if isinstance(augmentation['speed_perturbation_ratio'], list):
      stretch_amount = np.random.choice(augmentation['speed_perturbation_ratio'])
    elif augmentation['speed_perturbation_ratio'] > 0:
      # time stretch (might be slow)
      stretch_amount = 1.0 + (2.0 * np.random.rand() - 1.0) * \
                       augmentation['speed_perturbation_ratio']
    if stretch_amount > 0:
      signal_float = rs.resample(
          signal_float,
          sample_freq,
          int(sample_freq * stretch_amount),
          filter='kaiser_best',
      )

  # noise
  if 'noise_level_min' in augmentation and 'noise_level_max' in augmentation:
    noise_level_db = np.random.randint(low=augmentation['noise_level_min'],
                                       high=augmentation['noise_level_max'])
    signal_float += np.random.randn(signal_float.shape[0]) * \
                    10.0 ** (noise_level_db / 20.0)

  return signal_float


def preemphasis(signal, coeff=0.97):
  return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def get_speech_features(signal, sample_freq, params):
  """
  Get speech features using either librosa (recommended) or
  python_speech_features
  Args:
    signal (np.array): np.array containing raw audio signal
    sample_freq (float): sample rate of the signal
    params (dict): parameters of pre-processing
  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
    audio_duration (float): duration of the signal in seconds
  """

  backend = params.get('backend', 'psf')

  features_type = params.get('input_type', 'spectrogram')
  num_features = params['num_audio_features']
  window_size = params.get('window_size', 20e-3)
  window_stride = params.get('window_stride', 10e-3)
  augmentation = params.get('augmentation', None)

  if backend == 'librosa':
    window_fn = WINDOWS_FNS[params.get('window', "hanning")]
    dither = params.get('dither', 0.0)
    num_fft = params.get('num_fft', None)
    norm_per_feature = params.get('norm_per_feature', False)
    mel_basis = params.get('mel_basis', None)
    gain = params.get('gain')
    mean = params.get('features_mean')
    std_dev = params.get('features_std_dev')
    features, duration, signal = get_speech_features_librosa(
        signal, sample_freq, num_features, features_type,
        window_size, window_stride, augmentation, window_fn=window_fn,
        dither=dither, norm_per_feature=norm_per_feature, num_fft=num_fft,
        mel_basis=mel_basis, gain=gain, mean=mean, std_dev=std_dev
    )
  else:
    pad_to = params.get('pad_to', 8)
    features, duration = get_speech_features_psf(
        signal, sample_freq, num_features, pad_to, features_type,
        window_size, window_stride, augmentation
    )

  return features, duration, signal


def get_speech_features_librosa(signal, sample_freq, num_features,
                                features_type='spectrogram',
                                window_size=20e-3,
                                window_stride=10e-3,
                                augmentation=None,
                                window_fn=np.hanning,
                                num_fft=None,
                                dither=0.0,
                                norm_per_feature=False,
                                mel_basis=None,
                                gain=None,
                                mean=None,
                                std_dev=None):
  """Function to convert raw audio signal to numpy array of features.
  Backend: librosa
  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    num_features (int): number of speech features in frequency domain.
    pad_to (int): if specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    features_type (string): 'mfcc' or 'spectrogram'.
    window_size (float): size of analysis window in milli-seconds.
    window_stride (float): stride of analysis window in milli-seconds.
    augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`augment_audio_signal` for specification and example.

  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
    audio_duration (float): duration of the signal in seconds
  """
  signal = normalize_signal(signal.astype(np.float32), gain)
  signal, _ = librosa.effects.trim(signal, top_db=25)

  audio_duration = len(signal) * 1.0 / sample_freq

  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)
  num_fft = num_fft or 2**math.ceil(math.log2(window_size*sample_freq))

  S = np.abs(librosa.core.stft(preemphasis(signal,coeff=0.97), 
                               n_fft=num_fft,
                               hop_length=int(window_stride * sample_freq),
                               win_length=int(window_size * sample_freq),
                               center=True, window=window_fn))**2.0
  if mel_basis is None:
    # Build a Mel filter
    mel_basis = librosa.filters.mel(sample_freq, num_fft, n_mels=num_features,
                                    fmin=0, fmax=int(sample_freq/2))
  features = np.log(np.dot(mel_basis, S) + 1e-20).T

  norm_axis = 0 if norm_per_feature else None
  if mean is None:
    mean = np.mean(features, axis=norm_axis)
  if std_dev is None:
    std_dev = np.std(features, axis=norm_axis)

  features = (features - mean) / std_dev

  # now it is safe to pad
  # if pad_to > 0:
  #   if features.shape[0] % pad_to != 0:
  #     pad_size = pad_to - features.shape[0] % pad_to
  #     if pad_size != 0:
  #         features = np.pad(features, ((0,pad_size), (0,0)), mode='constant')
  return features, audio_duration, signal


def get_speech_features_psf(signal, sample_freq, num_features,
                            pad_to=8,
                            features_type='spectrogram',
                            window_size=20e-3,
                            window_stride=10e-3,
                            augmentation=None):
  """Function to convert raw audio signal to numpy array of features.
  Backend: python_speech_features
  Args:
    signal (np.array): np.array containing raw audio signal.
    sample_freq (float): frames per second.
    num_features (int): number of speech features in frequency domain.
    pad_to (int): if specified, the length will be padded to become divisible
        by ``pad_to`` parameter.
    features_type (string): 'mfcc' or 'spectrogram'.
    window_size (float): size of analysis window in milli-seconds.
    window_stride (float): stride of analysis window in milli-seconds.
    augmentation (dict, optional): dictionary of augmentation parameters. See
        :func:`augment_audio_signal` for specification and example.
    apply_window (bool): whether to apply Hann window for mfcc and logfbank.
        python_speech_features version should accept winfunc if it is True.
  Returns:
    np.array: np.array of audio features with shape=[num_time_steps,
    num_features].
    audio_duration (float): duration of the signal in seconds
  """
  if augmentation is not None:
    signal = augment_audio_signal(signal.astype(np.float32), 
        sample_freq, augmentation)
  signal = (normalize_signal(signal.astype(np.float32)) * 32767.0).astype(
      np.int16)

  audio_duration = len(signal) * 1.0 / sample_freq

  n_window_size = int(sample_freq * window_size)
  n_window_stride = int(sample_freq * window_stride)

  # making sure length of the audio is divisible by 8 (fp16 optimization)
  length = 1 + int(math.ceil(
      (1.0 * signal.shape[0] - n_window_size) / n_window_stride
  ))
  if pad_to > 0:
    if length % pad_to != 0:
      pad_size = (pad_to - length % pad_to) * n_window_stride
      signal = np.pad(signal, (0, pad_size), mode='constant')

  if features_type == 'spectrogram':
    frames = psf.sigproc.framesig(sig=signal,
                                  frame_len=n_window_size,
                                  frame_step=n_window_stride,
                                  winfunc=np.hanning)

    # features = np.log1p(psf.sigproc.powspec(frames, NFFT=N_window_size))
    features = psf.sigproc.logpowspec(frames, NFFT=n_window_size)
    assert num_features <= n_window_size // 2 + 1, \
      "num_features for spectrogram should be <= (sample_freq * window_size // 2 + 1)"

    # cut high frequency part
    features = features[:, :num_features]

  elif features_type == 'mfcc':
    features = psf.mfcc(signal=signal,
                        samplerate=sample_freq,
                        winlen=window_size,
                        winstep=window_stride,
                        numcep=num_features,
                        nfilt=2 * num_features,
                        nfft=512,
                        lowfreq=0, highfreq=None,
                        preemph=0.97,
                        ceplifter=2 * num_features,
                        appendEnergy=False)

  elif features_type == 'logfbank':
    features = psf.logfbank(signal=signal,
                            samplerate=sample_freq,
                            winlen=window_size,
                            winstep=window_stride,
                            nfilt=num_features,
                            nfft=512,
                            lowfreq=0, highfreq=sample_freq / 2,
                            preemph=0.97)
  else:
    raise ValueError('Unknown features type: {}'.format(features_type))

  if pad_to > 0:
    assert features.shape[0] % pad_to == 0
  mean = np.mean(features)
  std_dev = np.std(features)
  features = (features - mean) / std_dev

  return features, audio_duration

