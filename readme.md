# Localizing Anatomical Landmarks in Ocular Images using Zoom-In Attentive Networks

This is a reference implementation of our paper that will appear at the OMIA9 (MICCAI 2022) workshop.

Localizing Anatomical Landmarks in Ocular Images using Zoom-In Attentive Networks


![alt text](https://github.com/leixiaofeng-astar/OMIA9-ZIAN/blob/main/images/sacof_arch.png)

In this paper, we propose a zoom-in attentive network (ZIAN) for anatomical landmark localization in ocular images.
Input image is down-sampled and fed into the coarse network to get the per-pixel coarse heat-map. The Multi-ROIs centered at the peak pixel on the coarse heat-map, are cropped on the original input image, and fed into the fine network to build their feature representations. Next, Multi-ROIs features are refined by the co-attention module. Finally, multi-ROIs features are concatenated with coarse-level features, processed by self-attention module to get the fine heat-map.

# Quick start
## Installation
1. The code is developed under Python3.8 and Pytorch 1.9.0. Other versions of environment should work well, but have not been fully tested. Virtualenv is highly recommended for installation.
2. Install dependencies: 
   ```
   pip install -r requirements.txt
   ```
3. Pre-trained model download:

3.1 This ZIAN model uses the official HRNet project (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), Download HRNET pretrained models from Baidu Yun Drive: https://pan.baidu.com/s/1xucVbfCvkXSTu62b8NuN8Q (password: 6gpu), and put them into models/pretrained/.

4. ZIAN stated trained model download:

4.1 Download ZIAN models trained on REFUGE dataset (https://refuge.grand-challenge.org) from either https://www.dropbox.com/s/hkuhgy3sudaxsre/model_best_L1038_TL907_hrnet_SATA.pth?dl=0 or Baidu Yun Drive: https://pan.baidu.com/s/TBD (password: xxxx), and put them into output/refuge/fovea_net/refuge/;

4.2 Download ZIAN models trained on AGE dataset (https://age.grand-challenge.org) from either https://www.dropbox.com/s/eyttz30cbzzjs15/model_best_L960_FL14135_dsflipFL13638_hrnet_sata_LR1e4.pth?dl=0 or Baidu Yun Drive: https://pan.baidu.com/s/TBD (password: xxxx), and put them into output/refuge/fovea_net/refuge/.

5. Download the REFUGE data and uncompress them into a single directory, including the training, validation, and testing set. The folder structure should be like (by default, we use "sata-data" as ${DATA_ROOT}):
```
    ${DATA_ROOT}
        ├── AGE-test
        ├── AGE-test-GT
        ├── AGE-train
        ├── AGE-train-GT
  
    ${DATA_ROOT}
        ├── refuge1-test
        ├── refuge1-test-GT
        ├── refuge1-train
        ├── refuge1-train-GT
        
    ├── OMIA9-ZIAN
    ├── sata-data
```
5.1 Download REFUGE dataset (https://refuge.grand-challenge.org) from either https://www.dropbox.com/s/bmzx1h1byiexn6v/Dataset-Refuge.zip?dl=0 or Baidu Yun Drive: https://pan.baidu.com/s/TBD (password: xxxx), and put them into output/refuge/fovea_net/refuge/;

5.2 Download AGE dataset (https://refuge.grand-challenge.org) from Baidu Yun Drive: https://pan.baidu.com/s/15dd7eYDJf1H8nZ7ZA3BC_Q (password: b232), and put them into output/refuge/fovea_net/refuge/.


## Testing
Testing the pretrained model:
```
ZIAN with HRNET for Fovea localication in retina fundus image
python3 tools/test.py --cfg experiments/refuge.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best_L1038_TL907_hrnet_SATA.pth MODEL.SELF_ATTEN True MODEL.TRIP_ROI True MODEL.CO_ATTEN True MODEL.HRNET_TYPE 0

ZIAN with HRNET for scleral spur localization in AS-OCT images
python3 tools/test.py --cfg experiments/refuge-age.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best_L960_FL14135_dsflipFL13638_hrnet_sata_LR1e4.pth MODEL.SELF_ATTEN True MODEL.TRIP_ROI True MODEL.CO_ATTEN True TEST.FLIP_TEST True MODEL.HRNET_TYPE 0

The running log is as below:
(base) OMIA9-ZIAN$ python3 tools/test.py --cfg experiments/refuge.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best_L1038_TL907_hrnet_SATA.pth MODEL.SELF_ATTEN True MODEL.TRIP_ROI True MODEL.CO_ATTEN True MODEL.HRNET_TYPE 0
=> creating output/refuge/fovea_net/refuge
=> creating log/refuge/fovea_net/refuge_2022-07-30-19-22
Namespace(cfg='experiments/refuge.yaml', dataDir='', logDir='', modelDir='', opts=['TEST.MODEL_FILE', 'output/refuge/fovea_net/refuge/model_best_L1038_TL907_hrnet_SATA.pth', 'MODEL.SELF_ATTEN', 'True', 'MODEL.TRIP_ROI', 'True', 'MODEL.CO_ATTEN', 'True', 'MODEL.HRNET_TYPE', '0'], prevModelDir='')
...
Test: [1/50]	Time 1.962 (1.962)	LRInit 8.1675 (8.1675)	HRInit 7.8498 (7.8498)	Final 7.8498 (7.8498)	
Test: [21/50]	Time 0.238 (0.332)	LRInit 5.2200 (6.5387)	HRInit 4.6820 (6.2721)	Final 4.6820 (6.2721)	
Test: [41/50]	Time 0.223 (0.295)	LRInit 6.6190 (6.3817)	HRInit 7.1447 (6.1534)	Final 7.1447 (6.1534)	
SDR 5px:0.3675; 10px:0.76; 15px: 0.8775; 20px: 0.9225; L2:9.504680378735065
SDR 5px:0.4425; 10px:0.785; 15px: 0.89; 20px: 0.925; L2:9.07330446884036
SDR 5px:0.4425; 10px:0.785; 15px: 0.89; 20px: 0.925; L2:9.07330446884036
Average L2 Distance on evaluation: lr_L2 = 9.50, hr_L2 = 9.07
```

## Training
First ensure that the data root in experiments/refuge.yaml is set correctly. Note that the final performance may slightly differ from the pretrained model due to the randomness in the algorithm.
```
ZIAN with HRNET for Fovea localication
python3 tools/train.py --cfg experiments/refuge.yaml MODEL.SELF_ATTEN True MODEL.TRIP_ROI True MODEL.CO_ATTEN True MODEL.HRNET_TYPE 0 TRAIN.LR 0.0002 TRAIN.END_EPOCH 140 TRAIN.BATCH_SIZE_PER_GPU 4

ZIAN with HRNET for scleral spur localization in AS-OCT images
python3 tools/train.py --cfg experiments/refuge-age.yaml MODEL.SELF_ATTEN True MODEL.TRIP_ROI True MODEL.CO_ATTEN True MODEL.HRNET_TYPE 0 TRAIN.LR 0.0006 TRAIN.END_EPOCH 140 TRAIN.BATCH_SIZE_PER_GPU 4
```

## If you find this repository useful, please cite our paper:
 
    @misc{lei2022zian,
    title={Localizing Anatomical Landmarks in Ocular Images using Zoom-In Attentive Networks},
    author={Xiaofeng Lei, Shaohua Li, Xinxing Xu, Huazhu Fu, Yong Liu, Yih-Chung Tham, Yangqin Feng, Mingrui Tan, Yanyu Xu, Jocelyn Hui Lin Goh, Rick Siow Mong Goh, and Ching-Yu Cheng},
    year={2022},
    }
