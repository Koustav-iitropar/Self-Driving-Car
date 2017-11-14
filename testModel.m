function [Y]=testModel(X, W, FeatureSize)

W0=reshape(W(1:FeatureSize(2)*(FeatureSize(1)+1)),FeatureSize(2),FeatureSize(1)+1);
W1=reshape(W(FeatureSize(2)*(FeatureSize(1)+1)+1:FeatureSize(2)*(FeatureSize(1)+1)+FeatureSize(3)*(FeatureSize(2)+1)),FeatureSize(3),FeatureSize(2)+1);
W2=reshape(W(FeatureSize(2)*(FeatureSize(1)+1)+FeatureSize(3)*(FeatureSize(2)+1)+1:end),FeatureSize(4),FeatureSize(3)+1);

Y=[ones(size(X,1),1) sigmoid([ones(size(X,1),1) sigmoid(X*W0')]*W1')]*W2';
end
