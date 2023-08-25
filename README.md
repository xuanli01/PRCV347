# Enhancing Feature Representation for Anomaly Detection via Local-and-Global Temporal Relations and a Multi-Stage Memory

PaperID-347 for PRCV 2023

## Code Environment
```bash
conda env create -f environment.yaml
```
## Data Preparation

### ShanghaiTech and UCF-Crime
You can refer to the work  'Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning' [ICCV 2021]  https://github.com/tianyu0207/RTFM#setup for I3D features of ShanghaiTech and UCF-Crime datasets.

## Inference

### ShanghaiTech

First, unzip the compressed files in the folder 'sh_ckpt' into the checkpoint file 'sh_model.pkl'

```bash
zip -s 0 sh_ckpt/sh_model.zip --out sh_ckpt/ckpt.zip
unzip sh_ckpt/ckpt.zip 
```

Then, test the model

```python
python test.py -d shanghai -p /path/of/the/root/of/test/I3D/features -c sh_model.pkl
```
### UCF-Crime

First, unzip the compressed files in the folder 'ucf_ckpt' into the checkpoint file 'ucf_model.pkl'

```bash
zip -s 0 ucf_ckpt/ucf_model.zip --out ucf_ckpt/ckpt.zip
unzip ucf_ckpt/ckpt.zip
```
Then, test the model
```python
python test.py -d ucf -p /path/of/the/root/of/test/I3D/features -c ucf_model.pkl
```

## Supplementary Material
Supplementary material can be found in 0347_supp.pdf


