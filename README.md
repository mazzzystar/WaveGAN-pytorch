# WaveGAN-pytorch
PyTorch implementation of [Synthesizing Audio with Generative Adversarial Networks(Chris Donahue, Feb 2018)](https://arxiv.org/abs/1802.04208).

Befor running, make sure you have the `sc09Wav` dataset, and put that dataset under your current filepath.

## Quick Start:
1. Installation
```
sudo apt-get install libav-tools
```

2. Download dataset
* `sc09Wav`: [sc09 raw WAV files](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz), utterances of spoken english words '0'-'9'
* `piano`: [Piano raw WAV files](http://deepyeti.ucsd.edu/cdonahue/mancini_piano.tar.gz)

3. Run
```
# Make sure sc09Wav dataset under your current project filepath.
python train.py
```

#### Training time
* For `SC09Wav` dataset, 4 X Tesla P40 takes nearly 2 days to get reasonable result.
* For `piano` piano dataset, 2 X Tesla P40 takes 3-6 hours to get reasonable result.
* Decrease the `BATCH_SIZE` from 64 to 16 can acquire faster gradient descent but longer per-epoch time on multiple-GPU.

## Results
Generated "0-9": https://soundcloud.com/mazzzystar/sets/dcgan-sc09

Generated piano: https://soundcloud.com/mazzzystar/sets/wavegan-piano

Loss curve:

![](imgs/loss_curve.png)

## Architecture
![](imgs/archi.png)

## TODO
* [ ] Add some evaluation experiments, eg. inception score.

## Contributions
This repo is based on [chrisdonahue's](https://github.com/chrisdonahue/wavegan) and [jtcramer's](https://github.com/jtcramer/wavegan) implementation.
