# AI-algos

The repo contains two main sections. One focuses on machine learning algortithms and the other on deep learning algorithms. All the important algorithms are writen from scratch (primary dependency is numpy). 

## Machine learning algorithms from scratch

Each algorithm has its own file with the same class name. Most of the mathematical analysis is done using numpy functions. Each file contains a driver code to provide a demo of its working. Datasets provided by the scikit-learn library (`sklearn.datasets`) are used for running these demos.
The algorithms coded here are organized in the following manner (directory/filename)

```
                                        Supervised Learning
                                                 │   
                                        _________|_________
                                        |                 |
                                Classification        Regression
                                 |- Naive Bayes        |- Linear Reg.
                                 |- Logistic Reg.      |- Random Forest
                                 |- Decision Tree
                                 |- K-nearest neighbor
                                 |- Support Vector Classifier
```

```
                                       Unsupervised Learning
                                                 │   
                                        _________|_________
                                        |                 |
                                Clustering            Dimension Reduction  
                                |- K-means             |- Principal Component Analysis
                                |- K-modes             |- Discriminant Analysis
                                |- Hierarchical 
```

## Deep learning algorithms from scratch

There is a simple (shallow) neural network and a deep L-layer network. Helper functions, such as activation functions, cross-entropy formula etc., are in the utils.py file. 

---
Note: These files are for educational purposes only and can be copied, used for free. Some of the files in the ML section were already posted [here](https://www.kaggle.com/milan400/machine-learning-algorithms-from-scratch/notebook) on Kaggle. I will continue adding more algorithms with time.
