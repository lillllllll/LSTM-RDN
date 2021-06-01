imdsTrain = imageDatastore('stanfordcar\trainD\', ...
 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

imdsTest = imageDatastore('stanfordcar\testD\', ...
 'IncludeSubfolders',true, ...
 'LabelSource','foldernames');

targetSize = [224,224];
% targetSize = [227,227];
%  targetSize = [299,299];

imdsTrainA = augmentedImageDatastore(targetSize,imdsTrain);
imdsTestA = augmentedImageDatastore(targetSize,imdsTest);

% featureLayer='fc7'
% featureLayer='relu7'
featureLayer='pool5'
% featureLayer='avgpool2d'
% featureLayer='fc'
% featureLayer='avg_pool'%intercev2
%  featureLayer='efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'


 
load('convnetResnet18CarsX.mat')
trainFeatures = activations(convnet, imdsTrainA, featureLayer, 'MiniBatchSize',1024);
trainFeatures=squeeze(trainFeatures)';
save('Resnet18trainFeatures2.mat','trainFeatures','-v7.3')
%%
testFeatures = activations(convnet, imdsTestA, featureLayer, 'MiniBatchSize',1024);
testFeatures=squeeze(testFeatures)';
save('Resnet18testFeatures2.mat','testFeatures','-v7.3')
%%
num=size(imdsTest.Labels,1);numT=size(imdsTrain.Labels,1); 
fftest=gpuArray(testFeatures);
fftrain=gpuArray(trainFeatures);% 
fftest=zscore(fftest,1,2);
fftrain=zscore(fftrain,1,2);
% eudMatrix=sqrt(complex(repmat(sum(fftest.^2,2),1,numT)+repmat(sum(fftrain.^2,2),1,num)'-2*fftest*fftrain'));
eudMatrix=sqrt(abs(repmat(sum(fftest.^2,2),1,numT)+repmat(sum(fftrain.^2,2),1,num)'-2*fftest*fftrain'));
tt=gather(eudMatrix);

num=size(tt,1)
 
[~,indexx]=min(tt,[],2);
classMatrix=zeros(num,1);
testLabels=imdsTest.Labels;trainLabels=imdsTrain.Labels;
for i=1:num
 if testLabels(i)==trainLabels(indexx(i))
     classMatrix(i)=1;
 end
end

rate=sum(classMatrix)*100/num


