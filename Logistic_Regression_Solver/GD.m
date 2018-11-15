% Gradient descent
function [w,loss,iteration] = GD()
global x % n-1 * m
global y % 1 * m
global lamda
global u
n = size(x,1)+1;
m = length(y);
xd = [x;ones(1,m)];
w = 0.1*ones(1,n); % default w 1*n
loss = [];
for i = 1:10000
    u = max(20/i,0.01);
    fx = sum((log(1+exp(w*(-repmat(y,n,1).*xd)))),2)/m + lamda/2*norm(w(1:end-1))^2;
    gradient = sum((-repmat(y,n,1).*xd)./repmat((1+exp(w*(repmat(y,n,1).*xd))),n,1),2)/m + lamda*[w(1:end-1)';0];
    v = w - u*gradient';
    Q = fx + gradient'*(v-w)' + norm(w-v)^2/2/u;
    fxv = sum((log(1+exp(v*(-repmat(y,n,1).*xd)))),2)/m + lamda/2*norm(v(1:end-1))^2;
    if Q < fxv
        while Q < fxv
            u = u /2;
            v = w - u*gradient';
            Q = fx + gradient'*(v-w)' + norm(w-v)^2/2/u;
            fxv = sum((log(1+exp(v*(-repmat(y,n,1).*xd)))),2)/m + lamda/2*norm(v(1:end-1))^2;
        end
    end
    if norm(gradient) <= 10^-4
        break
    end
    w = v;
    loss =[loss,fx];
end
iteration = length(loss);


    
    
    
    