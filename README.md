# Progressive Update Guided Interdependent Networks for Single Image Dehazing
This is the PyTorch implementation for our paper:

**Aupendu Kar, Sobhan Kanti Dhara, Debashis Sen, Prabir Kumar Biswas. Progressive Update Guided Interdependent Networks for Single Image Dehazing. [[Project Website]](https://aupendu.github.io/iterative-dehaze) [[PAPER]](https://arxiv.org/abs/2008.01701)

## Data Preparation (NR-Haze Dataset)
### Training data 

1.1 Download the Training Set from [Google Drive]()

1.2 Put the dataset in `your_data_path` as follows:
```
your_data_path
└── train
        ├── NR-Indoor
                    ├── Clean
                            ├── 1.png
                            ├── .....
                            └── 1349.png
                    └── Depth
                            ├── 1.mat
                            ├── .....
                            └── 1349.mat
        └── NR-Outdoor
                    ├── Clean
                            ├── 0002.png
                            ├── .....
                            └── 8961.png
                    └── Depth
                            ├── 0002.mat
                            ├── .....
                            └── 8961.mat
```                          
Import `dataloader.py` to generate train pairs during training.

### Validation and Test data
1.1 Download the Validation Set from [Google Drive]()
1.2 Download the 7 Test Sets from [Google Drive]()

## Contact
Aupendu Kar: mailtoaupendu[at]gmail[dot]com
