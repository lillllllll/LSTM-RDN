# LSTM-RDN
TransferLearningResNet.m  is a transfer learning program

ExtractFeatureResNet.m  is a program to extract the fearues of transfied network

lstm-rdn.m  is a feature selective tansform network based on LSTM and ReLU and Dropout

ExtractFeatureLSTM_RDN.m used to extract the feature by LSTM-RDN, which inputs is the one-dimensional features of backbone

convnetResnet18Cars.mat is the transfered resnet18 model on stanfordcard

Resnet18trainFeatures.mat  contains  the features of Resnet18 of train set (stanfordcar)

Resnet18testFeatures.mat  contains  the features of Resnet18 of test set (stanfordcar)

Resnet18lstm_rdn1000.mat is the pre-trained lstm-rdn model with 1000 nunber of hidden units of LSTM

stanfordcar database which consists of the test and train sets is available from  
https://pan.baidu.com/s/1IJlwYmgmxo3bg6gpNv4RcA, extracting code: wjdq 
