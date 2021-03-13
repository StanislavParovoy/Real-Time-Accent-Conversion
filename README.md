# Real Time Foreign Accent Conversion

## Converts English utterances to another accent in 5 seconds, no need of parallel data

## Important

This pipeline didn't yield high robustness I was expecting (heavy accents poorly recognised & converted); I am no longer working on this topic.

## **Dependency**

[TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR) 

[TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)

## **Architecture** 
My pipeline is as follows:

![model architecture](model_architecture.png?raw=true "architecture")

### Module 1: Accent-Invariant Feature from Acoustic Model (AM)

This is the part where speech data with different accents are mapped to the same latent space. You can get an AM by training an ASR model. Instead of using a full ASR followed by some TTS model, I used outputs of intermediate layers of pretrained ASR models. This idea is inspired by [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/abs/1904.08983), but deeper layers are needed to drop the accent.

### Module 2: Domain-Specific Inverse Acoustic Model (IAM)

Here we map from latent space back to target accent domain. The key idea is that if a generative model has only seen data from a particular distribution / domain (e.g. North Amercan English), then it can only generate samples similar to those in that distribution. This is achieved by training a vocoder on a single-speaker dataset with one target accent, conditioned on the accent-invariant features extracted by the ASR model from module 1. This idea is inspired by [A Universal Music Translation Network](https://arxiv.org/abs/1805.07848).

### Module 3: Voice Cloning (VC) *

By training on a single-accent dataset, the vocoder generates speech with target accent AND speaker identity from that dataset. A voice cloning model is expected to restore the original speaker information.

## Train your own model

### Step 0 (Data)

#### 0.0
Download data based on your use case.
For instance, to convert to North American English:

AM: LibriTTS + VCTK

IAM: LJSpeech 

VC: LibriTTS + VCTK

#### 0.1
Preprocess data to get a "_raw.npy" and "_mel.npy" for each ".wav" file in your dataset folder. This is needed for IAM and VC datasets. All models will share the same set of stft parameters in `preprocess/preprocess.yml`. Example usage:
```
python3 preprocess/preprocess.py \
  --config preprocess/preprocess.yml \
  --suffix *.wav \
  --dataset PATH_TO_DATASET
```

Create train / eval csv files for asr model, some examples could be found in `python3 preprocess/create_asr_tsv.py`.

### Step 1 (AM)

#### 1.1
Train a speech recognition model based on the Multireader approach from section 2.3 of GE2E paper, using a subword Conformer (TensorFlowASR implementation):

```
python3 acoustic_model/train_subword_conformer_multi.py \
  --config acoustic_model/subword_conformer.yml \
  --subwords acoustic_model/conformer.subwords \
  --train-dir PATH_TO_CSV \
  --reg-dir PATH_TO_CSV \
  --dev-dir PATH_TO_CSV \
```

### Step 2 (IAM)

#### 2.1 
Extract accent-invariant features for your IAM dataset, using the ASR model from step 1.

For Conformer, I used the output of ConformerEncoder (shape: [num_samples/hop_size/4, 144]). One may need to experiment on which layer to use for other ASR models (e.g. Jasper: `conv115`, shape [num_samples/hop_size/2, 768]).

```
python3 acoustic_model/extract_subword_conformer.py \
  --config acoustic_model/subword_conformer.yml \
  --saved PATH_TO_H5_MODEL \
  --dataset PATH_TO_IAM_DATASET \
```

This will create a "_conformer_enc16.npy" for each ".wav" file in dataset folder.

#### 2.2
Train a vocoder conditioned on the extracted features, using the multiband-melgan (TensorFlowTTS implementation):

```
python3 inverse_acoustic_model/train_iam_wav.py \
  --train_dir PATH_TO_IAM_DATASET \
  --dev_dir PATH_TO_IAM_DATASET \
  --outdir saved_iam \
  --config inverse_acoustic_model/iam_conformer.yaml \
  --audio-query *_raw.npy \
  --mel-query *_conformer_enc16.npy

# resume training from checkpoint

python3 inverse_acoustic_model/train_iam_wav.py \
  --train_dir PATH_TO_IAM_DATASET \
  --dev_dir PATH_TO_IAM_DATASET \
  --outdir saved_iam \
  --config inverse_acoustic_model/iam_conformer.yaml \
  --audio-query *_raw.npy \
  --mel-query *_conformer_enc16.npy
  --resume saved_iam/checkpoints/ckpt-200000
```

### Step 3 (VC) *

#### 3.1
Train a speaker verification model, then use its output as speaker embedding vector.

I used GE2E with a modified model architecture. Currently it only supports single gpu, although speed bottleneck is likely at disk read. 

```
python3 ge2e/train_ge2e.py \
  --dataset PATH_TO_SV_DATASETS \
  --save saved_ge2e \
  --config ge2e/ge2e.yml

# generate a speaker embedding vector with suffic "_gc.npy" for each "_mel.npy" file in the voice cloning dataset.

python3 ge2e/extract_ge2e.py \
  --dataset PATH_TO_VC_DATASET \
  --config ge2e/ge2e.yml \
  --restore saved_ge2e
```

#### 3.2
Train a voice cloning model, using VQ-VAE + speaker embedding vector, on top of the multiband-melgan:

```
python3 voice_cloning/train_vc.py \
  --train_dir PATH_TO_VC_DATASET \
  --dev_dir PATH_TO_VC_DATASET \
  --outdir saved_vc \
  --audio-query *_raw.npy \
  --mel-query *_mel.npy \
  --config voice_cloning/vc.yaml 

# resume training from checkpoint

python3 voice_cloning/train_vc.py \
  --train_dir PATH_TO_VC_DATASET \
  --dev_dir PATH_TO_VC_DATASET \
  --outdir saved_vc \
  --audio-query *_raw.npy \
  --mel-query *_mel.npy \
  --config voice_cloning/vc.yaml \
  --resume saved_vc/checkpoints/ckpt-200000
```

### Step 4 (Assemble)

Example:

```
python3 inference.py \
  --trim_silence \
  --source test.wav \
  -am PATH_TO_AM_H5 \
  -iam PATH_TO_IAM_H5 \
  -sv PATH_TO_SV_H5 \
  -vc PATH_TO_VC_H5
```

This will save both IAM output and final output.

### Possible Improvements

[] Integrate [SPICE](https://tfhub.dev/google/spice/2) with IAM training (2.2) so it loses less of the original cadence.

[] Train a LM on asr features to improve its accuracy (dealing with mispronouciation).

[] During inference, speaker embedding calculation should run in parallel to IAM.

### Notes

It took 0.66s / 1.24s to convert 1s / 5s of audio on my macbook pro (late 2013, 2.4Ghz dual core i5, 4gb ram). 

The AM is the performance and speed bottleneck of this foreign accent conversion pipeline. It is crucial to train with mixed data with source and target accents.

\* IAM and VC could be combined together with enough clean data (i.e. a multispeaker version of LJSpeech); the only change is to bias-add global condition in addition to asr features. I had acceptable results on British subset of VCTK but not on other accent subsets.

The IAM could also generate mel spectrogram instead of full resolution audio. Since mel spectrogram runs on low resolution and we're only 4x away from target resolution, even autoregressive approaches are in real time. I tried with wavernn, the results were noisier than the full-resolution iam. 

For VC, I had better results training on raw audio than the preemphasis - output - deemphasis pipeline from LPCnet.

### References

- [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/abs/1904.08983)
- [A Universal Music Translation Network](https://arxiv.org/abs/1805.07848)
- [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/abs/1806.04558)
- [Jasper: An End-to-End Convolutional Neural Acoustic Model
](https://arxiv.org/abs/1904.03288)
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [Multi-band MelGAN: Faster Waveform Generation for High-Quality Text-to-Speech](https://arxiv.org/abs/2005.05106)
- [Generalized End-to-End Loss for Speaker Verification](https://arxiv.org/abs/1710.10467)
- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/)
- [VCTK](https://datashare.is.ed.ac.uk/handle/10283/3443)
- [LibriTTS](http://www.openslr.org/60/)

### Acknowlegements

- [NVIDIA OpenSeq2Seq](https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/jasper.html)
- [TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR)
- [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS)
- [VQVAE 1](https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py)
- [VQVAE 2](https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb)
- [WaveRNN](https://github.com/bshall/ZeroSpeech)

### License

GNU General Public License v2.0
