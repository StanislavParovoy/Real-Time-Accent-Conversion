# Real Time Foreign Accent Conversion

## Converts English utterances to North American accent in 3 seconds

## **Samples**


## **Requirements**

Install [TensorFlowASR](https://github.com/TensorSpeech/TensorFlowASR) and [TensorFlowTTS](https://github.com/TensorSpeech/TensorFlowTTS) following their instructions.
Make sure you can import `tensorflow_tts` and `tensorflow_asr` in this directory.

## **Using pretrained models**

`python3 demo.py --audio`demo.wav

## **How it works** 
The key idea is that if a generative model has only seen data from a particular distribution / domain (e.g. North Amercan English), then it can only generate samples similar to those in that distribution.

My pipeline is as follows:

x = original_audio
\
f = acoustic_model(x)
\
a = inverse_acoustic_model(f)
\
s = speaker_embedding(x)
\
output = voice_cloning(a, s)

### Stage 1: Accent-Invariant Feature from Acoustic Model (AM)

You can get an acoustic model by training an asr model. The most intuitve accent-invariant feature would be text, however this will lose all cadence / prosody / tone / speaker information of the original audio. On the other hand, outputs of intermediate layers of pretrained ASR models preserve most of these features. This idea is inspired by [TTS Skins: Speaker Conversion via ASR](https://arxiv.org/abs/1904.08983), but deeper layers are needed to drop the accent.

### Stage 2: Domain-Specific Inverse Acoustic Model (IAM)

We need a vocoder that can only generate speech with a particular accent. This is achieved by training (overfitting) a vocoder on dataset containing only one accent, conditioned on the accent-invariant features extracted by the ASR model from earlier.

### Stage 3: Voice Cloning (VC)

By overfitting on a single-accent dataset, the vocoder generates speech with target accent AND speaker identity from that dataset. The original speaker information is restored with a voice cloning model.

## Train your own model

### Step 0 (Data)

Download data based on your use case.
For instance, for converting English with random accent into North American accent:
For AM: LibriTTS + AccentDB + VCTK (a lot of accents)
For IAM: LJSpeech (target North American accent)
For VC: LibriTTS (a lot of speakers)

### Step 1 (AM)

#### 1.1 
Train a speech recognition model. I used pretrained Jasper from NVIDIA / Conformer.

See OpenSeq2Seq and TensorFlowASR.

#### 1.2
Fine-tune the model on accented data. This is based on the Multireader approach from section 2.3 of GE2E paper.

### Step 2 (IAM)

#### 2.1 
Extract accent-invariant features for your IAM dataset, using the ASR model from 1.1.

For Jasper, use the output of layer `conv115`(shape: [num_samples/hop_size/2, 768]). For Conformer, use the output of ConformerEncoder (shape: [num_samples/hop_size/4, 144]). You may need to experiment on which layer to use for other ASR models.

To use pretrained Jasper, run:

`python3 AcousticModel/Jasper/process_for_iam.py --restore PATH_TO_YOUR_MODEL --dataset PATH_TO_YOUR_IAM_DATASET --suffix _jasper_conv115`.

This will create a "_wav.npy" file and "_jasper_conv115.npy" for each ".wav" file in dataset folder.

To use pretrained Conformer, run:

`python3 AcousticModel/Conformer/process_for_iam.py --restore PATH_TO_YOUR_MODEL --dataset PATH_TO_YOUR_IAM_DATASET --suffix _conformer_enc16`.

This will create a "_raw.npy" file and "_conformer_enc16.npy" for each ".wav" file in dataset folder.

#### 2.2
Train a vocoder conditioned on the extracted features.

I used the multiband-melgan from TensorSpeech/TensorFlowTTS. Run:

`python3 InverseAcousticModel/train.py --train_dir PATH_TO_IAM_DATASET --dev_dir PATH_TO_IAM_DATASET --outdir saved_iam`

`python3 InverseAcousticModel/train.py --train_dir PATH_TO_IAM_DATASET --dev_dir PATH_TO_IAM_DATASET --outdir saved_iam --resume saved_iam/checkpoints/ckpt-200000`

### Step 3 (VC)

#### 3.1
Train a speaker verification model. Alternatively you could use one-hot vectors or embedding lookups and skip to 3.2.

I used GE2E with a modified model architecture. Run:

`python3 VoiceCloning/GE2E/train.py --dataset PATH_TO_SV_DATASET --save saved_ge2e`

Then generate a speaker embedding vector for each utterance. Run:

`python3 VoiceCloning/GE2E/infer.py --dataset PATH_TO_VC_DATASET --restore saved_ge2e`

For every audio file in your dataset folder with suffix ".wav", this should create a corresponding "\_gc.npy" file, which will be used as the global condition for the vc model.

#### 3.2
Train a voice cloning model.

I used VQ-VAE with multiband-melgan from TensorFlowTTS, modified to include speaker embedding vector as global condition. Run:

`python3 VoiceCloning/train.py --train_dir PATH_TO_VC_DATASET --dev_dir data/am_dataset --outdir saved_am`

`python3 VoiceCloning/train.py --train_dir PATH_TO_VC_DATASET --dev_dir data/am_dataset --outdir saved_am --resume saved_am/checkpoints/ckpt-200000`

### Step 4 (Assemble)

TODO

### Improvements

[TODO] Integrate [SPICE](https://tfhub.dev/google/spice/2) with IAM training (2.2) so it loses less of the original cadence.

[TODO] Train a LM on asr features to improve its accuracy (dealing with mispronouciation).

[TODO] Speaker embedding and its weight / bias calculation should run in parallel to IAM.

### Notes

The AM (ASR) is the performance bottleneck of this foreign accent conversion pipeline. Make sure you train with data from your source domain in addition to utterances with common accents. 

ASR is a heavy task; it is better to be put on cloud.

Stage 2 and 3 could be combined together if you have enough clean data (i.e. a multispeaker version of LJSpeech); the only change is to bias-add global condition in addition to asr features. I tried with British subset of VCTK data and it worked reasonably well, but didn't have good results on other accent subsets from VCTK.

For stage 2 (IAM), instead of constructing full resolution audio, you could also construct mel spectrogram, which will then be fed to the VC model. Note that you will not be able to use multi-stft loss with this approach. 

This pipeline will not generalise well to dialect conversion (e.g. Cantonese to Mandarine).

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
- [VQVAE encoder](https://github.com/bshall/ZeroSpeech)

### License

idgaf
