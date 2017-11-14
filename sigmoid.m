function [sig]=sigmoid(z)
sig=zeros(size(z));
sig(z>=0)=exp(-z(z>=0));
sig(z>=0)=1./(1+sig(z>=0));
sig(z<0)=exp(z(z<0));
sig(z<0)=sig(z<0)./(1+sig(z<0));
end