% Homework 2_2_linear
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
C = [0.001 0.01 0.1 1 10];
C_vs_loss_01 = zeros(length(C),2);
C_vs_loss_hinge = zeros(length(C),2);
for k = 1:length(C)
% c vector.
prob.c = [zeros(size(x,2)+size(y,2),1);C(k)*ones(size(x,1),1)];

xy = [repmat(y',size(x,2),1).*x';y';eye(size(x,1))];

prob.qosubi = [1:size(x,2)]';
prob.qosubj = [1:size(x,2)]';
prob.qoval  = ones(size(x,2),1);

% a, the constraint matrix

prob.a = xy';

% Lower bounds of constraints.
prob.blc  = ones(size(x,1),1);

% Upper bounds of constraints.
prob.buc  = [];   % There are no bounds.

% Lower bounds of variables.
prob.blx  = [-inf*ones(1,size(x,2)+size(y,2)),zeros(1,size(x,1))]';

% Upper bounds of variables.
prob.bux = [];   % There are no bounds.

% Display classifier
[r,res] = mosekopt('minimize',prob);

% Display return code.
fprintf('Return code: %d\n',r);

% Display primal solution for the constraints.
res.sol.itr.xc'

% Display primal solution for the variables.
res.sol.itr.xx'
% Test
w = res.sol.itr.xx(1:65,1);
testh = [newx,ones(size(newx,1),1)]*w;
testhy = newy.*testh;
index = find(testhy<=0);
% 0-1 loss
C_vs_loss_01(k,1) = C(k);
C_vs_loss_01(k,2) = length(index)/length(newy);

% hinge loss
C_vs_loss_hinge(k,1) = C(k);
C_vs_loss_hinge(k,2) = sum(1-testhy(index))/length(newy);

end
C_vs_loss_01
C_vs_loss_hinge
