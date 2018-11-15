% Homework 2_2nonlinear
addpath 'C:\Users\thinkpad\Mosek\7\toolbox\r2013a'
javaaddpath('C:\Users\thinkpad\Mosek\7\tools\platform\win64x86\bin\mosekmatlab.jar')
clc;
load('optdigits.tra');
x = optdigits(:,1:64);
y = optdigits(:,65);
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
% define the C
C=[0.000001 0.001 100];
C_vs_loss_01 = zeros(length(C),2);
C_vs_loss_hinge = zeros(length(C),2);
for k = 1:length(C)
    C(k)
%quadratic Kernel
Kq = zeros(length(y),length(y));
for i = 1:length(y)
    for j=1:length(y)
        Kq(i,j)=(x(i,:)*x(j,:)'+1)^2;
    end
end
yij = zeros(length(y),length(y));
for i = 1:length(y)
    for j=1:length(y)
        yij(i,j)=y(i,1)*y(j,1);
    end
end
KQ = yij.*Kq;
KQ = KQ + 10^(-5)*eye(length(y),length(y));
q     = KQ; 
% Set up the linear part of the problem. 
c     = -ones(length(y),1); 
a     = y'; 
blc   = 0; 
buc   = 0; 
blx   = zeros(length(y),1); 
bux   = C(k)*ones(length(y),1); 
% Optimize the problem. 
[res] = mskqpopt(q,c,a,blc,buc,blx,bux);  
% Show the primal solution. 
elfa = res.sol.itr.xx';
index1 = find( res.sol.itr.xx > 10^-8  );
index2 = find( res.sol.itr.xx < C(k)-10^-8  );
index = intersect(index1,index2);
b = y(index(1),1) - (elfa.*(y)')*((x*((x(index(1),:))')+1).^2 );
% Test
para = elfa.*(y');
testh = zeros(length(newy),1);
for i = 1:length(newy)
    testh(i,1)= para*((x*((newx(i,:))')+1).^2 ) + b;
end
testhy = newy.*testh;
indext = find(testhy<=0);
% 0-1 loss
C_vs_loss_01(k,1) = C(k);
C_vs_loss_01(k,2) = length(indext)/length(newy);
 
% hinge loss
C_vs_loss_hinge(k,1) = C(k);
C_vs_loss_hinge(k,2) = sum(1-testhy(indext))/length(newy);
end
C_vs_loss_01
C_vs_loss_hinge
