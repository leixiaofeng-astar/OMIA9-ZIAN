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

4.2 Download ZIAN models trained on REFUGE dataset (https://refuge.grand-challenge.org) from Baidu Yun Drive: https://pan.baidu.com/s/TBD(password: xxxx), and put them into output/refuge/fovea_net/refuge/;
4.3 Download ZIAN models trained on AGE dataset (https://age.grand-challenge.org) from Baidu Yun Drive: https://pan.baidu.com/s/TBD (password: xxxx), and put them into output/refuge/fovea_net/refuge/.

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
5.1 Download REFUGE dataset (https://refuge.grand-challenge.org) from Baidu Yun Drive: https://pan.baidu.com/s/TBD(password: xxxx), and put them into output/refuge/fovea_net/refuge/;

5.2 Download AGE dataset (https://refuge.grand-challenge.org)from Baidu Yun Drive: https://pan.baidu.com/s/TBD (password: xxxx), and put them into output/refuge/fovea_net/refuge/.


## Testing
Testing the pretrained model:
```
ZIAN with HRNET for Fovea localication in retina fundus image
python3 tools/test.py --cfg experiments/refuge.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best_L1038_TL907_hrnet_SATA.pth MODEL.SELF_ATTEN True MODEL.TRIP_ROI True MODEL.CO_ATTEN True MODEL.HRNET_TYPE 0

ZIAN with HRNET for scleral spur localization in AS-OCT images
python3 tools/test.py --cfg experiments/refuge-age.yaml TEST.MODEL_FILE output/refuge/fovea_net/refuge/model_best_L960_FL14135_dsflipFL13638_hrnet_sata_LR1e4.pth MODEL.SELF_ATTEN True MODEL.TRIP_ROI True MODEL.CO_ATTEN True MODEL.HRNET_TYPE 0
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
