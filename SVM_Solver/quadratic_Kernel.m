% Homework 2_1_c_quadratic Kernel
addpath 'C:\Users\thinkpad\Mosek\7\toolbox\r2013a'
javaaddpath('C:\Users\thinkpad\Mosek\7\tools\platform\win64x86\bin\mosekmatlab.jar')
clc;
C = [2 5 10 100];
% input 30 xi,yi from ISE426 last homework with some changes
yi = [1	1	1	-1	1	1	-1	-1	1	1	-1	-1	1	1	1	1	1	1	-1	1	1	1	1	1	1	1	1	-1	1	1];
xi = [-0.0192	-0.0302	-0.117	0.4454	-0.7989	0.0935	0.2654	0.604	-0.6324	0.977	0.926	0.8055	0.3007	-0.2771	0.3782	-0.5911	-0.2501	-0.113	0.9353	-0.1272	-0.0244	0.2476	0.1555	-0.9507	-0.6986	-0.4293	-0.8917	0.1545	-0.823	-0.5885; 0.4565	-0.8531	-0.9854	0.3952	-0.2569	0.7398	0.3098	-0.0959	-0.9139	-0.4862	0.0075	-0.0103	0.9564	0.1357	0.68	-0.1808	0.4231	0.8032	-0.259	0.9856	0.778	0.7701	-0.8341	-1	-0.0473	-0.9466	0.2226	0.4526	0.7856	0.5231];
for k = 1:4;
% Plot points.
subplot(2,2,k)
x_1 = xi(1,:);
x_2 = xi(2,:);
scatter(x_1,x_2,'filled','cdata',yi);
hold on
xlabel('x1');
ylabel('x2');
str = ['C=', num2str(C(k))];
title(str);
%quadratic Kernel
Kq = zeros(30,30); %30*30
for i = 1:30
    for j=1:30
        Kq(i,j)=(xi(:,i)'*xi(:,j)+1)^2;
    end
end

yij = zeros(30,30);
for i = 1:30
    for j=1:30
        yij(i,j)=yi(1,i)*yi(1,j);
    end
end
KQ = yij.*Kq;
KQ = KQ + 10^(-14)*eye(30,30);
q     = KQ; 
 
% Set up the linear part of the problem. 
c     = -ones(30,1); 
a     = yi; 
blc   = 0; 
buc   = 0; 
blx   = zeros(30,1); 
bux   = C(k)*ones(30,1); 
 
% Optimize the problem. 
[res] = mskqpopt(q,c,a,blc,buc,blx,bux); 
 
% Show the primal solution. 
res.sol.itr.xx
elfa = res.sol.itr.xx';
index1 = find( res.sol.itr.xx > 10^-5  );
index2 = find( res.sol.itr.xx < C(k)-10^-5  );
index = intersect(index1,index2);
scatter(xi(1,index), xi(2,index),'h','r','filled');
b = yi(1,index(1)) - (elfa.*yi)*((x_1*xi(1,index(1))+x_2*xi(2,index(1))+1).^2)';
x1 = -1:0.01:1;
x2 = -1:0.01:1;
para = elfa.*yi;
[x1,x2] = meshgrid(x1,x2);
func = @(x,y) para*((x_1*x+x_2*y+1).^2)' + b;
z = arrayfun(func,x1,x2);
contour(x1,x2,z,[0 0],'b-');
contour(x1,x2,z,[-1 1],'b:');
% Classify new points from last 5 points in ISE426 last homework
x_new = [-0.6578 -0.2009	0.0965	0.3399	-0.0519; -0.766	-0.7598	-0.12	0	0.9419];
y_new = ones(1,5);
for i = 1:5
    if para*((x_1*x_new(1,i)+x_2*x_new(2,i)+1).^2)' + b > 0
        y_new(1,i) = 1;
    else
        y_new(1,i) = -1;
    end
end
scatter(x_new(1,1:5),x_new(2,1:5),'v','filled','cdata',y_new)
end

