# Multilabel Lung Disease Diagnosis from Chest X-Ray Images

This project was completed as part of UCSD Extension's Machine Learning Bootcamp. 

The data was obtained from this [link](https://www.kaggle.com/datasets/nih-chest-xrays/data) and is a collection of ~100,000 labeled Chest X-Ray Images. The true labels were obtained via natural language processing of radiology reports and is estimated to be >90% accurate. It is a multilabel classification problem, as a single patient can be diagnosed with mutliple lung complications.

As this is an image processing problem, deep learning algorithms (specifically Convolutional Neural Networks) naturally became the preferred tool. This repository contains the notebooks for (4) different rounds of hyperparameter tuning, with each notebook tackling a different portion of the CNN architecture. Tuning was performed on a subset of the full training data, and the best performing models were then trained on the full dataset.

A summary of the numerical results can be found in 'capstone-results.xlsx'. Details of the notebooks are as follows:

- Data exploration and data wrangling notebooks can be found in capstone-data-exploration.ipynb & data-wrangling-nih-chest-x-ray.ipynb, respectively. 

- Round 1 tuning (capstone-tuning-round1.ipynb) focused on different configurations of the VGG16 model (pre-trained on ImageNet). This notebook experiments with sample weights, weighted loss functions, data sampling, and different pooling operations.

- Round 2 tuning (capstone-tuning-round2.ipynb) focused on image standardization (e.g. samplewise vs. feature-wise)

- Round 3 tuning (capstone-tuning-round3.ipynb) began to experiment with larger architectures, specifically the Xception model. 

- Round 4 tuning (capstone-tuning-round4.ipynb) expanded on round 3 and experimented with different configurations of Xception, Inception, and DenseNet models.

- The two best performing models were trained on the full dataset in capstone-scaled-prototype.ipynb
