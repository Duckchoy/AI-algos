# MLAlgos
## Machine learning algorithms from scratch using Numpy

Each algorithm has its own file with the same class name. Most of the mathematical analysis is done using numpy functions. Each file contains a driver code to provide a demo of its working. Datasets provided by the scikit-learn library (`sklearn.datasets`) are used for running these demos.
The algorithms coded here are organized in the following manner (directory/filename)

```
                                        Supervised Learning
                                                 │   
                                        _________|_________
                                        |                 |
                                Classification        Regression
                                 |- Naive Bayes        |- Linear Reg.
                                 |- Logistic Reg       |- Random Forest
                                 |- Decision Tree
                                 |- K-nearest neighbor
                                 |- Support Vector Machine
```

```
                                       Unsupervised Learning
                                                 │   
                                        _________|_________
                                        |                 |
                                Clustering          Dimension Reduction  
                                |- K-means             |- Principal Component Analysis
                                |- K-modes             |- Discriminant Analysis
                                |- Hierarchical 
```

There are a few deep learning algorithms as well. E.g., simple (shallow) neural network. I will add more notebooks soon.

---
Note: These files are for educational purposes only and can be copied, used for free. Some of these files were already posted [here](https://www.kaggle.com/milan400/machine-learning-algorithms-from-scratch/notebook) on Kaggle. I will continue adding more algorithms with time.
