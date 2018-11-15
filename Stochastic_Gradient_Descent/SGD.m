% Stochastic Gradient descent
function [w_bar,loss,iteration] = SGD()
global x % n-1 * m
global y % 1 * m
global lambda
global a
global p
n = size(x,1)+1;
m = length(y);
xd = [x;ones(1,m)];
w = 0.0001*ones(1,n); % default w 1*n including bias
loss = [];
ite = 1000;
w_bar = zeros(1,n);
for i = 1:ite
    % Step 1
    rand('seed',12345+1000*i);
    Index_A = randsample(m,p);
    Ak_x = xd(:,Index_A);
    Ak_y = y(Index_A);
    % Step 2
    AkPlus_x = [];
    AkPlus_y = [];
    for j = 1:p
        if w*Ak_x(:,j)*Ak_y(j) < 1
            AkPlus_x = [AkPlus_x,Ak_x(:,j)];
            AkPlus_y = [AkPlus_y,Ak_y(j)];
        end
    end
    % Step 3
    u = a/lambda/i;
    % Step 4
    yx = AkPlus_y*AkPlus_x';
    if length(yx) == 0
        yx = zeros(1,n);
    end
    w_half = w - u*lambda*w + u/p*yx;
    % Step 5
    w = min(1,1/sqrt(lambda)/norm(w_half,2))*w_half;
    fx = sum(max(0,(1-w*(repmat(y,n,1).*xd))),2)/m + lambda/2*norm(w(1:end-1))^2
    loss = [loss,fx];
    i
    % choose average w in second half of iterations
    if i > ite/2
        w_bar = w_bar + w/(ite/2);
    end
end
iteration = length(loss);


    
    
    
    