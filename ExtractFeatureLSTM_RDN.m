train=load('Resnet18trainFeatures.mat')
test=load('Resnet18testFeatures.mat')
%load('Resnet18lstm_rdn.mat')
load('trainLabels.mat')
load('testLabels.mat')

featureLayer='dropout_3' 
 
trainFeatures = activations(convnet, train.trainFeatures', featureLayer, 'MiniBatchSize',1024);
trainFeatures=squeeze(trainFeatures{1})';
testFeatures = activations(convnet, test.testFeatures', featureLayer, 'MiniBatchSize',1024);
testFeatures=squeeze(testFeatures{1})';
%%
num=size(imdsTest.Labels,1);numT=size(imdsTrain.Labels,1); 
fftest=gpuArray(testFeatures);
fftrain=gpuArray(trainFeatures);% 
fftest=zscore(fftest,1,2);
fftrain=zscore(fftrain,1,2);
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


