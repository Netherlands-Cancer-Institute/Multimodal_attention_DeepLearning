# MDL-IIA: Multimodal deep learning with intra- and inter-modality attention modules

The multi-modal deep learning algorithm was developed to predict the molecular subtypes of breast cancer. This model was combined with the attention mechanism to create the final model (multi-modal deep learning with intra- and inter-modality attention modules: MDL-IIA)

### Notes:

* Multimodal deep learning
1. Models_MDL-IIA_4-category.py: Model details for predicting 4-category molecular subtypes.
2. Models_MDL-IIA_L-NL.py: Model details for distinguishing between Luminal disease and Non-Luminal disease.

* Others
1. Model details based on mammography or ultrasound only are also provided. Please see "Models_mammography.py" and "Models_ultrasound.py".

* Training and testing: "python  .../(python_file).py"


### Requirements:

* tensorflow-gpu 2.4.0
* numpy 1.19.2
* pandas 1.2.4
* scikit-image 0.18.1
* scikit-learn 0.24.2

### Multimodal_data:

Data structure form. CC, craniocaudal (mammography). MLO, mediolateral oblique (mammography). US, ultrasound.

```
.
└── Multimodal_data
    ├── train   
    │     ├── 00001_CC.png
    │     ├── 00001_MLO.png
    │     ├── 00001_US.png
    │     ├── 00002_CC.png
    │     ├── 00002_MLO.png
    │     ├── 00002_US.png
    |     └── ...  
    │
    └── test   
          ├── 00001_CC.png
          ├── 00001_MLO.png
          ├── 00001_US.png
          ├── 00002_CC.png
          ├── 00002_MLO.png
          ├── 00002_US.png
          └── ... 
```

### Model details:

![image](https://github.com/Netherlands-Cancer-Institute/Multimodal_attention_DeepLearning/blob/main/Figures/Model_details.png)
Note: Model details of MDL-IIA. a, the proposed multi-modal deep learning with intra- and inter-modality attention model. b, the structure of channel and spatial attention. C, channel. H, height. W, width. Q, query. K, key. V, value. MG, mammography. US, ultrasound. MLO, mediolateral oblique view. CC, craniocaudal view. GAP, global average pooling. FC, fully-connected layer. HER2-E, HER2-enriched. TN, triple-negative.

### Visualization
Please refer to this article for the result visualization.
[Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://openaccess.thecvf.com/content_ICCV_2017/papers/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.pdf)

### Citation
If this work is helpful for you, please cite our paper as follows:

```
T. Zhang et al. Predicting breast cancer types on and beyond molecular level in a multi-modal fashion. NPJ Breast Cancer (2023).
[DOI] (https://doi.org/10.1038/s41523-023-00517-2)
```

### Contact details
If you have any questions please contact us. 

Email: ritse.mann@radboudumc.nl (Ritse Mann); taotanjs@gmail.com (Tao Tan); t.zhang@nki.nl (Tianyu Zhang)

Links: [Netherlands Cancer Institute](https://www.nki.nl/), [Radboud University Medical Center](https://www.radboudumc.nl/en/patient-care) and [Maastricht University](https://www.maastrichtuniversity.nl/nl)

<img src="https://github.com/Netherlands-Cancer-Institute/Multimodal_attention_DeepLearning/blob/main/Figures/NKI.png" width="166.98" height="87.12"/> <img src="https://github.com/Netherlands-Cancer-Institute/Multimodal_attention_DeepLearning/blob/main/Figures/RadboudUMC.png" width="231" height="74.58"/> <img src="https://github.com/Netherlands-Cancer-Institute/RadioLOGIC_NLP/blob/main/Figure/Maastricht.png" width="237.6" height="74.844"/>  
