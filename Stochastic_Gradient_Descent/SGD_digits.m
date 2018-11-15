% Stochastic Gradient descent tested on digits
clear all
clc
global x % n-1 * m
global y % 1 * m
global lambda
global a
global p
lambda = 0.26; % computed by 1/(C*n)
% Load data and prepare for build model
load('optdigits.tra');
x = optdigits(:,1:64)';
y = optdigits(:,65)';
load('optdigits.tes');
newx = optdigits(:,1:64);
newy = optdigits(:,65);
for i = 1:length(y)
    if y(i) == 5
        y(i) = 1;
    else
        y(i) = -1;
    end
end
for i = 1:length(newy)
    if newy(i) == 5
        newy(i) = 1;
    else
        newy(i) = -1;
    end
end
% define the lambdaset
pset =  [1 10 30 100];
aset = [1 10 100];
a_p_loss_01 = zeros(2,length(pset),length(aset));
a_p_loss_hinge = zeros(2,length(pset),length(aset));
tic;
for k = 1:length(aset)
    for t = 1:length(pset)
        p = pset(t);
        a = aset(k);
        [w,loss,iteration] = SGD();
        % Test
        testh = [newx,ones(size(newx,1),1)]*w';
        testhy = newy.*testh;
        index = find(testhy <= 0);
        % 0-1 loss
        a_p_loss_01(1,t,k) = pset(t);
        a_p_loss_01(2,t,k) = length(index)/length(newy);
        % hinge loss
        a_p_loss_hinge(1,t,k) = pset(t);
        a_p_loss_hinge(2,t,k) = sum(-testhy(index))/length(newy);
    end
end
toc;
display('Show the loss for each a & p')
a_p_loss_01
a_p_loss_hinge




