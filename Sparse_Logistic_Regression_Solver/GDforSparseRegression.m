% Gradient descent
% First order methods for Sparse Logistic Regression
% never increase ?
function [w,loss,iteration,performance] = GDforSparseRegression()
global x % n-1 * m
global y % 1 * m
global lamda
global u
n = size(x,1)+1;
m = length(y);
xd = [x;ones(1,m)];
w = zeros(1,n); % default w 1*n
loss = [];
u = 1;
performance = 0;
for i = 1:2000
    fx = sum((log(1+exp(w*(-repmat(y,n,1).*xd)))),2)/m + lamda*norm(w(1:end-1),1)
    loss_gradient = sum((-repmat(y,n,1).*xd)./repmat((1+exp(w*(repmat(y,n,1).*xd))),n,1),2)/m;
    gradient = loss_gradient.*abs([sign(w(1:end-1))';1]) + lamda*[sign(w(1:end-1))';0] + sign(loss_gradient).*max(abs(loss_gradient) - lamda,0).*[(1 - abs(sign(w(1:end-1))'));0];
    % above, use sign(w) and abs(sign(w)) to switch on/off different parts
    % of gradient for different situation of w(i)
    v = shrink(w - u*gradient');
    Q = fx + gradient'*(v-w)' + norm(w-v)^2/2/u + lamda*norm(v(1:end-1),1);
    fxv = sum((log(1+exp(v*(-repmat(y,n,1).*xd)))),2)/m + lamda*norm(v(1:end-1),1);
    if Q < fxv
        while Q < fxv
            u = u /2;
            v = shrink(w - u*gradient');
            Q = fx + gradient'*(v-w)' + norm(w-v)^2/2/u + lamda*norm(v(1:end-1),1);
            fxv = sum((log(1+exp(v*(-repmat(y,n,1).*xd)))),2)/m + lamda*norm(v(1:end-1),1);
            performance = performance + 1;
        end
    end
    if norm(gradient,inf) <= lamda*(1 + 10^-4 )
        break
    end
    w = v;
    loss =[loss,fx];
    performance = performance + 1;
end
iteration = length(loss);


    
    
    
    