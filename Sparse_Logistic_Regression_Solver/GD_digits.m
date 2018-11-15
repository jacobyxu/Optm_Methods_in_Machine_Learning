% Gradient descent tested on digits
% First order methods for Sparse Logistic Regression
% never increase ?
clear all
clc
global x % n-1 * m
global y % 1 * m
global lamda
global u
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
% define the lamdaset
lamdaset =  [0.001 0.01 0.05 0.1 0.5];
lamda_vs_loss_01 = zeros(length(lamdaset),2);
lamda_vs_loss_hinge = zeros(length(lamdaset),2);
performance_time = zeros(3,length(lamdaset));
for k = 1:length(lamdaset)
    lamda = lamdaset(k);
    tic;
    [w,loss,iteration,performance] = GDforSparseRegression();
    lamda;
    loss;
    w;
    iteration;
    % Test
    testh = [newx,ones(size(newx,1),1)]*w';
    testhy = newy.*testh;
    index = find(testhy<=0);
    % 0-1 loss
    lamda_vs_loss_01(k,1) = lamdaset(k);
    lamda_vs_loss_01(k,2) = length(index)/length(newy);
    % hinge loss
    lamda_vs_loss_hinge(k,1) = lamdaset(k);
    lamda_vs_loss_hinge(k,2) = sum(1-testhy(index))/length(newy);
    toc;
    performance_time(3,k) = toc;
    performance_time(1,k) = performance;
    performance_time(2,k) = iteration;
end
display('Show the loss for each lamda')
lamda_vs_loss_01
lamda_vs_loss_hinge
display('Show the performance, iterate times and total time(s) for each lamda')
performance_time



