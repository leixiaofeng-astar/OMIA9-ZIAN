# Coding Test

The code structure is basically based on the official HRNet project (https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), with plenty of modifications.

Trained with the REFUGE 2018 train+val data, the best model achieves 7.5 pixels in average L2 distance on the REFUGE 2018 Test set.

## Quick start
### Installation
1. The code is developed under Python2.7 and Pytorch 0.4.0. Other versions of environment should work well, but have not been fully tested. Virtualenv is highly recommended for installation.
2. Install dependencies: 
   ```
   pip install -r requirements.txt
   ```
3. Initialize output (training model output directory) and log (tensorboard log directory) directory:
   ```
   mkdir output
   mkdir log
   ```
4. Download HRNET pretrained models from Baidu Yun Drive: https://pan.baidu.com/s/1xucVbfCvkXSTu62b8NuN8Q (password: 6gpu), and put them into models/pretrained.

5. Download the REFUGE data and uncompress them into a single directory, including the training, validation, and testing set. The folder structure should be like
```
    ${DATA_ROOT}
        ├── AGE-test
        ├── AGE-test-GT
        ├── AGE-train
        ├── AGE-train-GT
```

### Testing
Testing the pretrained model:
```
python tools/test.py --cfg experiments/refuge.yaml TEST.MODEL_FILE ./models/pretrained/best_model.pth
```

### Training
First ensure that the data root in experiments/refuge.yaml is set correctly. Note that the final performance may slightly differ from the pretrained model due to the randomness in the algorithm.
```
python tools/train.py --cfg experiments/refuge.yaml
```
