# MPNN_NMR

### NMR is the most import structural elucidation tool in chemical synthesis. Analysing NMR spectra can be assisted by simulating the expected spectrum for comparison. DFT is the common way of doing this but is an expensive and arduous computation. Machine learning is a much faster alternative. In Bristol, we have developed IMPRESSION [1] - a machine learning tool for predicting NMR chemical shifts and coupling constants.

![alt text]('git@github.com:Benhonore/MPNN_NMR.git/MPNN.png')

### This repository contains the code for training and testing the performance of popular message passing layers in the Pytorch Geometric [2] framework for predicting chemical shifts of 3D molecules. The two message passing layers used are: 

### - Graph Convolutional Network layer, published by Kipf and Welling [3] 
### - Neural Message Passing for Quantum Chemistry, published by Gilmer and Schoenholz [4].

### The representation is a molecular graph. This means the important distinction between using these two models is 2D versus 3D representation. In the case of the Neural Message Passing layer, internal distances between every atom (or node) in the molecule are included as edge features, making the representation 3D. For the GCN layer, the molecular graph is partially connected. 

##### 1.) W. Gerrard, L. A. Bratholm, M. Packer, A. J. Mulholland, D. R. Glowacki and C. P. Butts, Chem Sci, 2020, 11, 508-515.
##### 2.)
##### 3.)
##### 4.)
