
imdsTrain = imageDatastore('stanfordcar\trainD\', ...
 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

imdsTest = imageDatastore('stanfordcar\testD\', ...
 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

targetSize = [224,224];
imdsTrain = augmentedImageDatastore(targetSize,imdsTrain);

load('resnet18.mat');
% net=efficientnetb0;
% net=nasnetmobile
% net=inceptionresnetv2;
%plot(net)
featureLayer='pool5'
% featureLayer='avg_pool'
% featureLayer='global_average_pooling2d_2'
% featureLayer='efficientnet-b0'
lgraph = layerGraph(convnet); 
numClasses = 196
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);% 
%  lgraph = replaceLayer(lgraph,'predictions',newLearnableLayer);%inceptionresnetv2
 lgraph = replaceLayer(lgraph,'fc1000',newLearnableLayer);%101,18
% lgraph = replaceLayer(lgraph,'efficientnet-b0|model|head|dense|MatMul',newLearnableLayer);

newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);%resnet18,101
% lgraph = replaceLayer(lgraph,'classification',newClassLayer);%efficientb0
%%
options = trainingOptions('sgdm','MaxEpochs',10,...
	'InitialLearnRate',0.001,'ExecutionEnvironment','gpu','MiniBatchSize',20);  
%,'Shuffle','every-epoch'
% lgraph = layerGraph(convnet); %
convnet = trainNetwork(imdsTrain,lgraph,options);
save('convnetResnet18CarsX.mat','convnet')

imdsTest2 = augmentedImageDatastore(targetSize,imdsTest);  
YTest = classify(convnet,imdsTest2,'MiniBatchSize',50); 
accuracy = 100*sum(YTest == imdsTest.Labels)/numel(imdsTest.Labels)  

disp over



