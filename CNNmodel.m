classdef CNNmodel
    properties
        NumConvlayer=1;
        NumFullCon=1;
        kernels=cell(1,1);
        poolSize;
        KernelSize;
        alpha=0.1;
        W;
    end
    methods
        function obj=CNNmodel(inputlayerSize,numkernel,poolSize,KernelSize)
%            inputlayerSize
            obj.kernels{1}=rand(KernelSize(1),KernelSize(2),inputlayerSize(3),numkernel)*0.2-0.1;
            obj.poolSize=poolSize;
            obj.KernelSize=KernelSize;
            obj.W=rand(floor((inputlayerSize(1)-KernelSize(1)+1)/poolSize(1))*floor((inputlayerSize(2)-KernelSize(2)+1)/poolSize(2))*numkernel,1);
        end
        function [output,Clayer1]=testmodel(obj,inputlayer)
            n=size(inputlayer,4);
            Clayer1=struct('conv','relu','pool','out');
            Clayer1.conv=obj.conv2d(inputlayer);
            Clayer1.relu=obj.relu(Clayer1.conv);
            Clayer1.pool=obj.pool(Clayer1.relu);
            Clayer1.out=reshape(Clayer1.pool,size(obj.W,1),n)';
            output=obj.fullCon(Clayer1.out);
        end
        function [out]=conv2d(obj,input)
            n=size(input,4);
            [~,~,~,d]=size(obj.kernels{1});
            out=zeros(size(input,1)-obj.KernelSize(1)+1,size(input,2)-obj.KernelSize(2)+1,d,n);
            for i=1:n
                for k=1:d
%                    size(input)
%                    size(obj.kernels{1})
%                    conv=conv2(input(:,:,j,i),obj.kernels{1}(:,:,j,k),'same');
%                    out(:,:,k,i)=out(:,:,k,i)+conv;
                    out(:,:,k,i)=convn(input(:,:,:,i),obj.kernels{1}(:,:,:,k),'valid');
                    
                end
            end
        end
        function [out]=relu(obj,input)
            out=input;
            %Ids=find(out<0);
            out(out<0)=0; %out(Ids)=obj.alpha*out(Ids);
        end
        function [out]=pool(obj,input)
            out=zeros(floor(size(input)./[obj.poolSize,1,1]));
            for i=1:size(out,1)
                for j=1:size(out,2)
                    out(i,j,:,:)=max(max(input((i-1)*obj.poolSize(1)+1:i*obj.poolSize(1),(j-1)*obj.poolSize(2)+1:j*obj.poolSize(2),:,:)));
                end
            end
        end
        function [out]=fullCon(obj,input)
            out=input*obj.W;
        end
        function Train(obj,inputlayer,output,lrRate,batchSize,iteration)
            obj.W=rand(size(obj.W))*0.2-0.1;  % initialize full connected layer weights
            obj.kernels{1}=rand(size(obj.kernels{1}))*0.2-0.1;% initialize conv layer kernels
            N=size(inputlayer,4); % training ex
            batch=randperm(N); % batch
            for it=1:iteration
                for i=1:batchSize:N
                    range=i+batchSize-1;
                    if range>N
                        range=N;
                    end
                    X=inputlayer(:,:,:,batch(i:range)); % batch input
                    Y=output(batch(i:range)); % batch output

                    [Out,Clayer1]=obj.testmodel(X); % forward pass
                    err=(Out-Y)'*(Out-Y); % MS error
                    cost=err;
                    gredW=Clayer1.out'*(Out-Y)/size(Y,1); % gredient w.r.t. full-conn layer weights
                    gredK1=(Out-Y)*obj.W'; % gredient w.r.t. input of full-conn
                    gredK1=reshape(gredK1',size(Clayer1.pool)); % gredient w.r.t. output of pool
                    [a,b,~,~]=size(gredK1); % a*b image dim. of pool output image
                    gred=zeros(size(Clayer1.relu)); % gredient w.r.t. output of relu will be calc.
                    for i=1:a
                        for j=1:b % (i,j) a single pixel of pool
                            A=reshape(Clayer1.relu((i-1)*obj.poolSize(1)+1:i*obj.poolSize(1),...
                                (j-1)*obj.poolSize(2)+1:j*obj.poolSize(2),:,:),obj.poolSize(1)*...
                                obj.poolSize(1),size(Clayer1.relu,3),size(Clayer1.relu,4)); % A=2d patch corr. to pool(i,j,:,:)
                            [~,Mj]=max(A); % id of max(A) which goes to pool output
                            for k=1:size(Mj,2)
                                for l=1:size(Mj,3)
                                    gred((i-1)*obj.poolSize(1)+mod((Mj(1,k,l)-1),obj.poolSize(1))+1,...
                                        (j-1)*obj.poolSize(2)+floor((Mj(1,k,l)-1)/obj.poolSize(1))+1,k,l)...
                                        =gredK1(i,j,k,l); % gred w.r.t. to patch-max in relu output(pool input)
                                end
                            end
                        end
                    end
                    gred(Clayer1.conv<0)=0; % gred w.r.t. input of relu(conv. output)
                    G=zeros(size(obj.kernels{1}));
                    for k=1:size(G,3)
                        for l=1:size(G,4)
                            G(:,:,k,l)=convn(X(:,:,k,:),gred(:,:,l,:),'valid')/size(gred,4); % gredient w.r.t. kernel
                        end
                    end
%                    gred=[G(:);gredW(:)];
                    
                    
                    % update:::::
                    obj.kernels{1}=obj.kernels{1}-lrRate*G;
                    obj.W=obj.W-lrRate*gredW;
                end
                
                
                [Out,~]=obj.testmodel(inputlayer);
                cost=(Out-output)'*(Out-output);
                fprintf('iteration:%d cost:%f\n',it,cost);
            end
        end
        function [cost,gred]=CostCNN(obj,inputlayer,Y)
            [Out,Clayer1]=obj.testmodel(inputlayer);
            err=(Out-Y)'*(Out-Y);
            cost=err;
            gredW=Clayer1.out'*(Out-Y);
            gredK1=(Out-Y)*obj.W';
            gredK1=reshape(gredK1',size(Clayer1.pool));
            [a,b,~,~]=size(gredK1);
            gred=zeros(size(Clayer1.relu));
            for i=1:a
                for j=1:b
                    A=reshape(Clayer1.relu((i-1)*obj.poolSize(1)+1:i*obj.poolSize(1),...
                        (j-1)*obj.poolSize(2)+1:j*obj.poolSize(2),:,:),obj.poolSize(1)*...
                        obj.poolSize(1),size(Clayer1.relu,3),size(Clayer1.relu,4));
                    [~,Mj]=max(A);
                    for k=size(Mj,1)
                        for l=size(Mj,2)
                            gred((i-1)*obj.poolSize(1)+mod((Mj(k,l)-1),obj.poolSize(1))+1,...
                                (j-1)*obj.poolSize(2)+floor((Mj(k,l)-1)/obj.poolSize(1))+1,k,l)...
                                =gredK1(i,j,k,l);
                        end
                    end
                end
            end
            gred(Clayer1.conv<0)=0;
            G=zeros(size(obj.kernels{1}));
            for k=1:size(G,3)
                for l=1:size(G,4)
                    G(:,:,k,l)=convn(inputlayer(:,:,k,:),gred(:,:,l,:),'valid');
                end
            end
            gred=[G(:);gredW(:)];
            
            
            
            
            
%             h=0.00001;
%             [m,n,c,d]=size(obj.kernels{1});
%             gredK=zeros(m,n,c,d);
%             for i=1:m
%                 fprintf("i:%d\n",i);
%                 for j=1:n
%                     fprintf("j:%d\n",j);
%                     for k=1:c
%                         fprintf("k:%d\n",k);
%                         for l=1:d
%                             fprintf("l:%d\n",l);
%                             obj.kernels{1}(i,j,k,l)=obj.kernels{1}(i,j,k,l)+h;
%                             Out=obj.testmodel(inputlayer);
%                             Finerr=(Out-Y)'*(Out-Y);
%                             gredK(i,j,k,l)=(Finerr-err)/h;
%                             obj.kernels{1}(i,j,k,l)=obj.kernels{1}(i,j,k,l)-h;
%                         end
%                     end
%                 end
%             end
%             m=size(obj.W,1);
%             gredW=zeros(m,1);
%             for i=1:m
%                 obj.W(i)=obj.W(i)+h;
%                 Out=obj.testmodel(inputlayer);
%                 Finerr=(Out-Y)'*(Out-Y);
%                 gredW(i)=(Finerr-err)/h;
%                 obj.W(i)=obj.W(i)-h;
%             end
%             gred=[gredK(:);gredW(:)];
        end
    end
end