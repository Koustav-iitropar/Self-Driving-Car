%% data fetching
%function [] = l32a(alpha, batchSize, trainIter, dropOut)
inputlayerSize=[32,32,3,100];numkernel=32;poolSize=[4,4];KernelSize=[3,3];
Data=importdata('steering\data.txt');
Data.data(1)=[];
Data.textdata(1)=[];
Data.rowheaders(1)=[];
Perm=randperm(size(Data.data,1));
Y=Data.data(Perm);
Files=Data.textdata(Perm);
X=zeros(32,32,3,size(Y,1));
for i=1:size(Y,1)
    if mod(i,1000)==0
        disp(i)
    end
    X(:,:,:,i)=im2double(imread(['steering\' Files{i}(3:end)]));
end
D=X;

%% CNN
model=CNNmodel(inputlayerSize,numkernel,poolSize,KernelSize);
model.Train(X,Y,0.01,20,10);
%[cost,gred]=model.CostCNN(X(:,:,:,1:20),Y);
%end

%% dropout
X=zeros(32,32,size(Y,1));
for i=1:size(Y,1)
    X(:,:,i)=rgb2gray(D(:,:,:,i));
end
X=reshape(X,1024,size(Y,1))';
TrainX=X(Perm(1:floor(0.8*size(X,1))),:);
TrainX=[ones(size(TrainX,1),1) TrainX];
TrainY=Y(Perm(1:floor(0.8*size(X,1))));
TestX=X(Perm(floor(0.8*size(X,1))+1:end),:);
TestX=[ones(size(TestX,1),1) TestX];
TestY=Y(Perm(floor(0.8*size(X,1))+1:end));

FeatureSize=[1024,512,64,1];
[W]=trainModel(TrainX,TrainY,TestX,TestY,0.001,32,1000,0.1,FeatureSize);
TestO=testModel(TestX,W,FeatureSize);
fprintf('Testerror:%f\n',(TestO-TestY)'*(TestO-TestY)/size(TestO,1));