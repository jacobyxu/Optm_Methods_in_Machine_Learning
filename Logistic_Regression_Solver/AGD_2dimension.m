% Accelerated gradient descent tested on 2-demensional data set
clear all
clc
global x % n-1 * m
global y % 1 * m
global lamda
tic;
% input 30 xi,yi from ISE426 last homework
y = [1	-1	-1	1	1	1	1	-1	-1	-1	-1	-1	1	1	1	1	1	1	-1	1	1	1	-1	-1	1	-1	1	1	1	1];
x = [-0.0192	-0.0302	-0.117	0.4454	-0.7989	0.0935	0.2654	0.604	-0.6324	0.977	0.926	0.8055	0.3007	-0.2771	0.3782	-0.5911	-0.2501	-0.113	0.9353	-0.1272	-0.0244	0.2476	0.1555	-0.9507	-0.6986	-0.4293	-0.8917	0.1545	-0.823	-0.5885; 0.4565	-0.8531	-0.9854	0.3952	-0.2569	0.7398	0.3098	-0.0959	-0.9139	-0.4862	0.0075	-0.0103	0.9564	0.1357	0.68	-0.1808	0.4231	0.8032	-0.259	0.9856	0.778	0.7701	-0.8341	-1	-0.0473	-0.9466	0.2226	0.4526	0.7856	0.5231];
% Plot points.
subplot(2,1,1)
x1 = x(1,:);
x2 = x(2,:);
scatter(x1,x2,'filled','cdata',y);
hold on
xlabel('x1');
ylabel('x2');
title('linearly separable training sets and 5 tests');
lamda = 10^-3;
[w,loss,iteration] = AGD();
loss
w
w1 = w(1,1);
w2 = w(1,2);
b = w(1,3);
xp = -1:0.000001:1;
yp = -w1/w2*xp - b/w2;
plot(xp,yp)
% Plot new points
x_new = [-0.6578 -0.2009	0.0965	0.3399	-0.0519; -0.766	-0.7598	-0.8755	0.8383	0.9419];
y_new = ones(1,5);
for i = 1:5
    if x_new(1,i)*w1 + x_new(2,i)*w2 + b > 0
        y_new(1,i) = 1;
    else
        y_new(1,i) = -1;
    end
end
scatter(x_new(1,1:5),x_new(2,1:5),'v','filled','cdata',y_new)
% Plot loss function in each iteration
subplot(2,1,2)
line(1:length(loss),loss)
xlabel('iteration');
ylabel('loss');
title('loss vs. iteration');
toc;

