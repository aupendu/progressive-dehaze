# Progressive Update Guided Interdependent Networks for Single Image Dehazing
This is the PyTorch implementation for our paper:

**Aupendu Kar, Sobhan Kanti Dhara, Debashis Sen, Prabir Kumar Biswas. Progressive Update Guided Interdependent Networks for Single Image Dehazing. [[Project Website]](https://aupendu.github.io/iterative-dehaze) [[PAPER]](https://arxiv.org/abs/2008.01701)

## Test Codes
To perform Image Dehazing
``
python test.py --datatype real --modelx IPUDN_IHaze \
--testfolder realdata
``

* Give test image set path through ``--testfolder`` argument
* Select model through ``--modelx`` argument
* Select data type through ``--datatype`` argument, ``real`` for Real and ``synthetic`` for Synthetic Haze data

## Contact
Aupendu Kar: mailtoaupendu[at]gmail[dot]com
