# Progressive Update Guided Interdependent Networks for Single Image Dehazing
This is the PyTorch implementation for our paper:

**Aupendu Kar, Sobhan Kanti Dhara, Debashis Sen, Prabir Kumar Biswas. Progressive Update Guided Interdependent Networks for Single Image Dehazing. [[Project Website]](https://aupendu.github.io/iterative-dehaze) [[PAPER]](https://arxiv.org/abs/2008.01701)

## Data Preparation (NR-Haze Dataset)
### Training data 

1.1 Download the Training Set from [Google Drive](https://drive.google.com/file/d/1r0fKCZ5TP2cRPK-XBOFJCiIpYiUxmbkL/view?usp=sharing)

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

### Validation data
1.1 Download the Validation Set from [Google Drive](https://drive.google.com/drive/folders/1peVM1RclTgD7-KXf6bR3R9NHtOcmy96v?usp=sharing) and put it in `data` folder


## Training Codes

Train Transmission Map Estimation Model
```
python main.py --train_t --loss 1*SSIM\
                --transmodel Transmission  \
                --batch_size 16 --test_every 2000 --epochs 250 \
                --lr 1e-4 --gamma 0.5 --decay 50+100+150+200 \
                --exp_name EX1 --patch_size 224 \
                --trainfolder NR-Indoor+NR-Outdoor --b_min 0.2+2.0 --b_max 0.8+5.0 \
                --valset NR_Indoor_Outdoor --valfolder NR-Indoor+NR-Outdoor \
                --b_minVal  0.2+2.0 --b_maxVal 0.8+5.0 \
                --A_vary 40 --A_min 0.3 --A_max 1.0
```

Train Atmospheric Light Estimation Model
```
python main.py --train_a --loss 1*MSE \
                --atmmodel Atmospheric \
                --batch_size 16 --test_every 2000 --epochs 250 \
                --lr 1e-4 --gamma 0.5 --decay 50+100+150+200 \
                --exp_name EX1 --patch_size 224 \
                --trainfolder NR-Indoor+NR-Outdoor --b_min 0.2+2.0 --b_max 0.8+5.0 \
                --valset NR_Indoor_Outdoor --valfolder NR-Indoor+NR-Outdoor \
                --b_minVal  0.2+2.0 --b_maxVal 0.8+5.0 \
                --A_vary 40 --A_min 0.3 --A_max 1.0
```
Train Dehazing Model
```
python main.py --train_h --loss 1*L1 \
                --transmodel Transmission --transmodel_pt model_dir/EX1_Transmission_1*SSIM_bestSSIM.pth.tar \
                --atmmodel Atmospheric --atmmodel_pt model_dir/EX1_Atmospheric_1*MSE_bestMSE.pth.tar \
                --hazemodel IPUDN_IHaze --iter 6 \
                --batch_size 2 --test_every 2000 --epochs 500 \
                --lr 1e-4 --gamma 0.5 --decay 100+200+300+400 \
                --exp_name EX1 --patch_size 224 \
                --trainfolder NR-Indoor+NR-Outdoor --b_min 0.2+2.0 --b_max 0.8+5.0 \
                --valset NR_Indoor_Outdoor --valfolder NR-Indoor+NR-Outdoor \
                --b_minVal  0.2+2.0 --b_maxVal 0.8+5.0 \
                --A_vary 40 --A_min 0.3 --A_max 1.0
```
Fine-tune the whole framework together
```
python finetune.py
```



## Contact
Aupendu Kar: mailtoaupendu[at]gmail[dot]com
