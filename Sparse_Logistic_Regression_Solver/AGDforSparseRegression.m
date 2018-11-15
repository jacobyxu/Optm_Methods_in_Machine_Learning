% Accelerated gradient descent
% First order methods for Sparse Logistic Regression
function [w,loss,iteration,performance] = AGDforSparseRegression()
global x % n-1 * m
global y % 1 * m
global lamda
global u
n = size(x,1)+1;
m = length(y);
xd = [x;ones(1,m)];
w = ones(1,n);% default w 1*n
v = ones(1,n);% default w 1*n
loss = [];
t = 0;
tplus = (1+sqrt(1+4*t^2))/2;
r = (1-t)/tplus;
t = tplus;
u = 1;
performance = 0;
for i = 1:2000
    fw = sum((log(1+exp(w*(-repmat(y,n,1).*xd)))),2)/m + lamda*norm(w(1:end-1),1);
    loss_gradient = sum((-repmat(y,n,1).*xd)./repmat((1+exp(w*(repmat(y,n,1).*xd))),n,1),2)/m;
    gradient = loss_gradient.*[abs(sign(w(1:end-1))');1] + lamda*[sign(w(1:end-1))';0] + sign(loss_gradient).*max(abs(loss_gradient) - lamda,0).*[(1 - abs(sign(w(1:end-1))'));0];
    % above, use sign(w) and abs(sign(w)) to switch on/off different parts
    % of gradient for different situation of w(i)
    wplus = shrink(v - u*gradient');
    fv = sum((log(1+exp(v*(-repmat(y,n,1).*xd)))),2)/m + lamda*norm(v(1:end-1),1);
    Q = fv + gradient'*(wplus-v)' + norm(v-wplus)^2/2/u + lamda*norm(v(1:end-1),1);
    fwplus = sum((log(1+exp(wplus*(-repmat(y,n,1).*xd)))),2)/m + lamda*norm(wplus(1:end-1),1);
    if Q < fwplus
        while Q < fwplus
            u = u /2;
            wplus = shrink(v - u*gradient');
            fv = sum((log(1+exp(v*(-repmat(y,n,1).*xd)))),2)/m + lamda*norm(v(1:end-1),1);
            Q = fv + gradient'*(wplus-v)' + norm(v-wplus)^2/2/u + lamda*norm(v(1:end-1),1);
            fwplus = sum((log(1+exp(wplus*(-repmat(y,n,1).*xd)))),2)/m + lamda*norm(wplus(1:end-1),1);
            performance = performance + 1;
        end
    end
    if norm(gradient,inf) <= lamda + 10^-2
        break
    end
    vplus = (1-r)*wplus + r*w;
    tplus = (1+sqrt(1+4*t^2))/2;
    r = (1-t)/tplus;
    t = tplus;
    w = wplus;
    v = vplus;
    loss =[loss,fw];
    performance = performance + 1;
end
iteration = length(loss);
