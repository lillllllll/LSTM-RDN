  
load('testLabels.mat')
load('trainLabels.mat')
load('Resnet18testFeatures2.mat')
load('Resnet18trainFeatures2.mat') 

numberF=1000;
tFeatures=[trainFeatures];
teFeatures=[testFeatures];

fn=size(teFeatures,2)
numHiddenUnits =1200;
numClasses = 196;
maxEpochs = 1000;
miniBatchSize =512;
layers = [ 
    sequenceInputLayer(fn)           
    dropoutLayer(0.5)
    lstmLayer(numHiddenUnits) 
    reluLayer
    dropoutLayer(0.5)
    %reluLayer
    dropoutLayer(0.5)
    
    fullyConnectedLayer(numClasses,'name','f3') %9658
    softmaxLayer
    classificationLayer
    ];
  
options = trainingOptions('rmsprop', ...
    'ExecutionEnvironment','gpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',1200, ...
     'InitialLearnRate',0.0003, ...
    'MiniBatchSize',256, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch');
testLabels2=categorical(testLabels)';trainLabels2=categorical(trainLabels)';
convnet = trainNetwork(tFeatures',trainLabels2,layers,options) 
% save('resnet18lstm.mat','convnet')
 
YPred = classify(convnet,teFeatures','MiniBatchSize',miniBatchSize);
 
acc = 100*sum(YPred == testLabels2)./numel(testLabels2)
% save('res18_1acc','acc')

