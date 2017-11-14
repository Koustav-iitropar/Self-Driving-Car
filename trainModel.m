function [W]=trainModel(X, Y, TestX, TestY, alpha, batchSize, iter, dropOut, FeatureSize)

N=size(X,1);
batch=randperm(N);
Trainerror=zeros(iter,1);
Testerror=zeros(iter,1);

W0=rand(FeatureSize(2),FeatureSize(1)+1)*0.02-0.01;
W1=rand(FeatureSize(3),FeatureSize(2)+1)*0.02-0.01;
W2=rand(FeatureSize(4),FeatureSize(3)+1)*0.02-0.01;
W0(:,1)=0;
W1(:,1)=0;
W2(:,1)=0;

for it=1:iter
    for i=1:batchSize:N
        range=i+batchSize-1;
        if range>N
            range=N;
        end
        l1DropOut=randperm(size(X,2)-1);
        l1DropOut=l1DropOut(1:floor(dropOut*size(X,2)-1));
        l2DropOut=randperm(size(W1,1)-1);
        l2DropOut=l2DropOut(1:floor(dropOut*size(W1,1)-1));
        l3DropOut=randperm(size(W2,1)-1);
        l3DropOut=l3DropOut(1:floor(dropOut*size(W2,1)-1));
        H1=[ones(range-i+1,1) sigmoid(X(batch(i:range),:)*W0')];
        H2=[ones(range-i+1,1) sigmoid(H1*W1')];
        O=H2*W2';

        gred3=O-Y(batch(i:range));
        gred2=(gred3*W2(:,2:end)).*H2(:,2:end).*(1-H2(:,2:end));
        gred2(:,l3DropOut)=0;
        gred1=(gred2*W1(:,2:end)).*H1(:,2:end).*(1-H1(:,2:end));
        gred1(:,l2DropOut)=0;
        D3=gred3'*H2;
        D2=gred2'*H1;
        D1=gred1'*X(batch(i:range),:);
        D1(:,l1DropOut)=0;
        D2(:,l2DropOut)=0;
        D3(:,l3DropOut)=0;
        W2=W2-alpha*D3/batchSize;
        W1=W1-alpha*D2/batchSize;
        W0=W0-alpha*D1/batchSize;
    end
    W=[W0(:);W1(:);W2(:)];
    O=testModel(X,W,FeatureSize);
    TestO=testModel(TestX,W,FeatureSize);
    Trainerror(it)=(O-Y)'*(O-Y)/size(O,1);
    Testerror(it)=(TestO-TestY)'*(TestO-TestY)/size(TestO,1);
    fprintf('iteration:%d, error:%f\n',it,(O-Y)'*(O-Y)/size(O,1));
end
plot(1:iter,Trainerror,'b-');
hold on;
plot(1:iter,Testerror,'r-');

end

