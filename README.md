# SSL-using-PaCMAP

## Table of Contents
1. [Overview](#overview)
2. [Data](#data)
3. [PaCMAP](#pacmap)
4. [Motivation of the algorithm](#motivation-of-the-algorithm)
5. [Method](#method)  
   5.1 [Data split](#data-split)  
   5.2 [Training](#training)  
   5.3 [Classification](#classification)  
6. [Experiments](#experiments)
7. [Results](#results)
8. [Suggested future experiments](#suggested-future-experiments)  
    8.1 [Transfer learning](#transfer-learning)  
    8.2 [Sensitivity analysis](#sensitivity-analysis)  
    8.3 [Fine tuning](#fine-tuning)  
    8.4 [Other supervised algorithms](#other-supervised-algorithms)  
    8.5 [Combining with other SSL techniques](#combining-with-other-ssl-techniques)
9. [Limitations](#limitations)  
   9.1 [Randomness](#randomness)  
   9.2 [Architecture choices](#architecture-choices)  
10. [Attributions](#attributions)



### Overview
This repository explores using the unsupervised dimensionality reduction algorithm PaCMAP to make use of unlabeled data in computer vision tasks.

### Data
The data used is the MNIST data-set which is a set of 60 000 pictures of handwritten digits along with its corresponding label (a number 0-9). The pictures are 28 by 28 pixels with values ranging from 0-255 depending on the pixel intensity. The data-set is commonly used to test and benchmark computer vision techniques. In these experiments the labels are removed for some fractions of the data which means one can't use them to train the model in a supervised way.

### PaCMAP
[PaCMAP (Pairwise Controlled Manifold Approximation Projection)](https://dl.acm.org/doi/abs/10.1145/2751205.2751225) is an algorithm used for dimensionality reduction designed to preserve both the local and global structure of high-dimensional data when embedding it into a lower-dimensional space. It works orks by optimizing distances between pairs of points in the lower dimensional space, using the distance classes:
Nearby points: Preserving local structure.
Distant points: Preserving global structure.
Mid-range points: Maintaining a balance between local and global representations.

The algorithm is most commonly used to reduce data into 2d or 3d in order to allow for visualization. In this project it is instead used to reduce to a much higher dimensional space. The hope is that it can utilize unlabeled data to get a low dimensional approximation of the manifold which hand-written digits is on which can then allow for much more data-efficient supervised learning.

### Motivation of the algorithm
I believe this way of utilizing PaCMAP in semi-supervised learning could be beneficial because:
1. It should be able to help find the manifold each data-class exists on in a rather compute-efficient manner.
2. It can make use of unlabeled data in a way which should aid supervised learning in the later steps of the training.
3. Training only a sub-network instead of training the whole network at once in each phase should make the task more memory efficient and potentially faster.
4. Training only a few layers at a time should avoid common problems with training deep networks such as vanishing gradients.

### Method

##### Data split
The data is split into labeled and unlabeled given a ratio. For the unlabeled set the label is removed. 

##### Training
The unlabeled data is reduced in dimensionality (in this case from 784 dimensions to 64 dimension) by applying the PaCMAP algorithm. One "problem" with the PaCMAP algorithm is that it only places the data-points in the lower dimensional space, it doesn't give a transformation between the high- and low-dimensional space. To side-step this an MLP (which will be referred to as net1) is trained taking the original, high-dimensional, unlabeled data as input and PaCMAP<sub>64</sub>(unlabeled data) as labels. This allows us to approximately transform any datapoint into this lower-dimensional space, even if it was not used in the original PaCMAP. 

The unlabeled data points are then passed through net1 and then PaCMAP is applied again, compressing from 64 dimensions to 16 dimensions. A second MLP (which will be referred to as net2) is then trained by taking net<sub>1</sub>(unlabeled data) as input and PaCMAP<sub>16</sub>(net<sub>1</sub>(unlabeled data)) as the labels.

In the last step the labeled data is passed through net1 and then net2. A third MLP (net3) is then trained using the net<sub>2</sub>(net<sub>1</sub>(labeled data)) as input and their corresponding label as the targets. 

#### Classification
Classifying data points is done by taking argmax(net<sub>3</sub>(net<sub>2</sub>(net<sub>1</sub>(data)))).

### Experiments
All models were tested using different fractions of the data labeled. The tested fractions were: 50%, 10%, 5%, 1%, 0.1% and one labeled sample per class.

1. Base model: A base model was created using the approach described in the method (see base_model.ipynb).
2. Early stopping: A model where the MLPs are trained using early stopping. When the MLPs were trained 10% of the data was used as validation data, as soon as validation loss increased training stopped. For the one sample per class case early stopping was not used in the supervised phase as that would otherwise lead to one class having no labeled data. Number of epochs was in this case exchanged for max_epochs which had a default value of 20.
3. Labeled as unlabeled: In these experiments the labeled data was used in both the unsupervised parts (PaCMAP and training of net1 and net2) and in the supervised part.
4. Concatenated1: In the last supervised stage all the MLPs were converted to one MLP. This was done by training net1 and net2 as previously. Then net3 is initialized, the three models are concatenated and this concatenated model is trained using the pictures as input and the labels as outputs.
5. Concatenated2: When net2 is initialized, it is concatenated with net1 and then trained using the unlabeled data as input and PaCMAP<sub>16</sub>(net<sub>1</sub>(unlabeled data)) as the targets. When net3 is initialized it's concatenated with net1 and net2 lite in the concatenated1 experiments.
6. All combined: This experiment combined early stopping, labeled as unlabeled and concatenated1.

### Results
|  Test accuracy | 50% | 10% | 5% | 1% | 0.1% | 1 sample per class|
|----|----|----|----|----|----|----|
| Base model | 0.9526 | 0.9602 | 0.9571 | 0.9489 | 0.4107* | 0.2783* |
| Early stopping | 0.9436 | 0.9582 | 0.9546 | 0.9573 | 0.6235 | 0.2184* |
| Labeled as unlabeled | 0.9596 | 0.9597 | 0.9553 | 0.9463 | 0.1079* | 0.0989* |
| Concatenated1 | 0.9752 | 0.9587 | 0.9582 | 0.9449 | 0.678 | 0.5499 |
| Concatenated2 | 0.9735 | 0.9613 | 0.9593 | 0.9513 | 0.6117 | 0.2667 |
| All combined | 0.9684 | 0.9619 | 0.9506 | 0.936 | 0.677 | 0.2825 | 

\* Seems to be very random, sometimes these values indicate a random classification (accuracy around 0.1) and sometimes they reach significantly higher.

### Suggested future experiments

#### Transfer learning
Maybe one could utilize some pretrained model, take the first layer(s) of that model and then utilize this approach on the embeddings produced by the pretrained model. I believe this can significantly enhance the model, especially if the pretrained model is a CNN/ViT. The reason I believe this is that this these models are less sensitive about where the object is located, now clustering using PaCMAP is limited since a picture and the exact same picture shifted on pixel to the right are very far apart in the input space which is used for the initial PaCMAP embeddings.

#### Sensitivity analysis
It would be interesting to try to better understand why some cases are so sensitive to randomness and how to mitigate it. One idea is to try to force labeled data points belonging to different classes to be far away from each other in the PaCMAP embeddings. It would also be interesting to do each experiment many times in order to get a range of what accuracies to expect and with what certainties.

#### Fine Tuning
One thind I believe could benefit the models is fine tuning. Fine Tuning could be done with network architecture, learning rate and learning rate schedulers, number of epochs used etc. This would likely increase the model's performance by a few percent. 

#### Other supervised algorithms
Since PaCMAP is an algorithm whose goal is to preserve local structure and does some clustering it would be interesting to try whether other, simpler algorithms can be used in the supervised part of training and classification (instead of net3). It would be interesting to see how well KNNs or decision trees would perform.

#### Combining with other SSL techniques
It would be very interesting to combine this with other techniques such as pseudo-labeling and consistency regularization. Psuedo-labeling can, if successful, increase the labeled data-set which should increase performance. Consistency regularization encourage the network to make similar predictions for data-points which are similar to each other. This should help increase performance, especially if PaCMAP does "great" clustering in it's dimensionality reduction.

### Limitations

#### Randomness
Many results, especially those with very little labeled data give differing test results between different training runs. I believe the main source of randomness is the train-test-split. I believe that if the labeled data is for some reason not very representative of its class then this will result in very poor generalization. I also believe the clustering PaCMAP does in its dimensionality reduction is what makes the successful runs relatively successful and the successful runs so bad. If the labeled data-points behave nicely and different classes are located in different clusters then the mapping to the class prediction becomes straightforward, even if it only has a few data points. On the other hand, if only a few labeled data points are "impostors" in other clusters then training the training of net3 will produce very complex separators for the classes which will lead to pretty much random classification except for data very close to the labeled data. 

#### Architecture choices
Each experiment is done with the same architecture. It may be smarter to choose the architecture based on the amount of data. For example when only one sample per class is available then the PaCMAP maybe should be done more aggressively in order to make the job as easy as possible for net3, the current approach is instead heavily over-paremterized. On the other hand, when there's plenty of labeled data then one maybe instead want to relax the PaCMAP and let the labeled data do more of the heavy lifting.

### Attributions
This project is based on an idea I had myself, however, I don't know if the idea is original and I don't claim it is. The code in the project is written by me, Claude 3.5 Sonnet and github CoPilot.

