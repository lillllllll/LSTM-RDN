# LSTM-RDN
TransferLearningResNet.m  is a transfer learning program

ExtractFeatureResNet.m  is a program to extract the fearues of transfied network

lstm-rdn.m  is a feature selective tansform network based on LSTM and ReLU and Dropout

ExtractFeatureLSTM_RDN.m used to extract the feature by LSTM-RDN, which inputs is the one-dimensional features of backbone

convnetResnet18Cars.mat is the transfered resnet18 model on stanfordcard

Resnet18trainFeatures.mat  contains  the features of Resnet18 of train set (stanfordcar)

Resnet18testFeatures.mat  contains  the features of Resnet18 of test set (stanfordcar)

Resnet18lstm_rdn1000.mat is the pre-trained lstm-rdn model with 1000 nunber of hidden units of LSTM

stanford-cars database which consists of the test and train sets is available from  
https://pan.baidu.com/s/1IJlwYmgmxo3bg6gpNv4RcA, extracting code: wjdq 

# usage
step1, TransferLearningResNet.m is used to perform the transfer learning with ResNet18 backbone on stanford-cars.   
       Resnet18trainFeatures and Resnet18testFeatures will be produced.
       
step2, Training lstm-rdn model by lstm-rdn.m on Resnet18trainFeatures. Resnet18lstm_rdn1000 will be produced.

step3, Computing the classification accurancy with Resnet18lstm_rdn1000 on Resnet18testFeatures.

step4(optinal), Extracting the trainFeature and testFeature and perform final evaluation by using ExtractFeatureLSTM_RDN.m. 

# Citations
@article{chaorong2021,  
author = {li chaorong and Yuanyuan Huang and WEI HUANG and Fengqing Qin},  
title = {One-dimensional DCNN Feature Selective Transformation with LSTM-RDN for Image Classification},  
year = {2021},  
month = "6",  
url = {https://www.techrxiv.org/articles/preprint/One-dimensional_DCNN_Feature_Selective_Transformation_with_LSTM-RDN_for_Image_Classification/14642814},  
doi = {10.36227/techrxiv.14642814.v2}  
}
