## Lung Disease Diagnosis using Deep Learning
This project was completed as part of the [UCSD Machine Learning Engineering Bootcamp](https://career-bootcamp.extension.ucsd.edu/programs/machine-learning-engineering/). 

The final product (model) of this project has also been deployed as a [web app](https://tungtngyn-lung-disease-diagnosis-app-app-5j0fvw.streamlitapp.com/).


### Project Description
The goal of this project is to identify lung diseases from chest x-ray images. The dataset comes from [this Kaggle repository](https://www.kaggle.com/datasets/nih-chest-xrays/data) and comprises of 100K+ chest x-ray images from 30K+ unique patients. This is a multi-label problem, meaning each chest x-ray image can have multiple diseases associated with it. The då†aset is also heavily imbalanced, both with respects to the labels (the majority of the labels are negative) as well as the classes (some classes only contain ~100 samples vs. other classes with 10000+). The primary learning objective of this project was centered around deep learning for image classification.


### Technologies & Methods Used
* scikit-learn for misc. ML utilities
* pandas & numpy for data analysis
* TensorFlow, specifically Keras, for model prototyping
* Transfer Learning using pre-trained VGG16, Xception, and DenseNet models
* streamlit for deployment


### Project Directory
./exploratory-data-analysis.ipynb
* Preliminary exploration of the dataset.

./model-tuning-1.ipynb
* Model prototypes using a subset of the data & a pre-trained VGG16 model.
* Tested different variations of VGG16, data sampling approaches, and weighted loss functions.

./model-tuning-2.ipynb
* Model prototypes using a subset of the data & pre-trained Xception and DenseNet models.
* Repea†ed the same studies as in the first model tuning notebook.

./scaled-prototype.ipynb
* Trained the best performing architecture from model-tuning-1 & model-tuning-2 on the full dataset.
* Also contains a threshold selection study as well as final metrics on the test dataset.

./results/
* ROC, PR curves for all models


### Summary of Results
Due to the imbalance in the dataset, the area under the precision-recall curve (AUPR) was selected as the primary evaluation metric for making decisions regarding model architecture. Main conclusions gained from this project were as follows:

* The best performing model with respects to average AUPR was a pre-trained Xception model.

* Oversampling the minority classes and using a weighted loss function did not perform well for any pre-trained model.

* The larger the model, the better it performed, even if only slightly.

The AUPR (averaged over all labels) on the test dataset for the best performing model (model 8b - Xception) was 0.258. The PR curve is shown below.

[image](./results/test-data-pr-model-8b.jpg)


### Areas for Improvement & Further Experimentation

* Tweaking the classification threshold for each label - Choosing the default classification threshold of 0.5 leads to poor behavior in Consolidation and Pneumonia, but potentially acceptable behavior (depending on use-case) for all the other labels. This approach would need to be rigorously studied as it can result in overfitting to the training/validation data.

* More preprocessing of images - Parsing through the chest x-ray images, one can see the large amount of noise within the images themselves. Some images are zoomed in, some are taken from the side instead of front/back, and others have reference notes added by the radiologist which obscure part of the chest. Further standardization of the images is needed for this dataset.

* Scaling up and/or different architectures - there exists larger models than Xception and DenseNet169. Going from VGG16 from Xception improved the metrics for all labels, thus, it is reasonable to believe scaling up improves accuracy (but at the cost of more training time).

* Improving true labels - the labels from this dataset are obtained using NLP, which means some of the true labels are actually false. While the maintainers of the data predict a 90% accuracy for the true labels, this still means around ~10,000 images are incorrectly labeled. Correcting the true labels will help reduce noise.

* If computational resources are available, using a OneVsRest scheme and training one classifier for each label.