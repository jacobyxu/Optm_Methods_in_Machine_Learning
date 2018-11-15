% Accelerated gradient descent
function [w,loss,iteration] = AGD()
global x % n-1 * m
global y % 1 * m
global lamda
global u
n = size(x,1)+1;
m = length(y);
xd = [x;ones(1,m)];
w = 0.1*ones(1,n);% default w 1*n
v = 0.1*ones(1,n);% default w 1*n
loss = [];
t = 0;
tplus = (1+sqrt(1+4*t^2))/2;
r = (1-t)/tplus;
t = tplus;
for i = 1:10000
    u = max(20/i,0.01);
    fw = sum((log(1+exp(w*(-repmat(y,n,1).*xd)))),2)/m + lamda/2*norm(w(1:end-1))^2;
    gradient = sum((-repmat(y,n,1).*xd)./repmat((1+exp(v*(repmat(y,n,1).*xd))),n,1),2)/m + lamda*[v(1:end-1)';0];
    wplus = v - u*gradient';
    fv = sum((log(1+exp(v*(-repmat(y,n,1).*xd)))),2)/m + lamda/2*norm(v(1:end-1))^2;
    Q = fv + gradient'*(wplus-v)' + norm(v-wplus)^2/2/u;
    fwplus = sum((log(1+exp(wplus*(-repmat(y,n,1).*xd)))),2)/m + lamda/2*norm(wplus(1:end-1))^2;
    if Q < fwplus
        while Q < fwplus
            u = u /2;
            wplus = v - u*gradient';
            fv = sum((log(1+exp(v*(-repmat(y,n,1).*xd)))),2)/m + lamda/2*norm(v(1:end-1))^2;
            Q = fv + gradient'*(wplus-v)' + norm(v-wplus)^2/2/u;
            fwplus = sum((log(1+exp(wplus*(-repmat(y,n,1).*xd)))),2)/m + lamda/2*norm(wplus(1:end-1))^2;
        end
    end
    if norm(gradient) <= 10^-4
        break
    end
    vplus = (1-r)*wplus + r*w;
    tplus = (1+sqrt(1+4*t^2))/2;
    r = (1-t)/tplus;
    t = tplus;
    w = wplus;
    v = vplus;
    loss =[loss,fw];
end
iteration = length(loss);
