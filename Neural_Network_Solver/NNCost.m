function [loss,Theta_1,Theta_2] = NNCost(nn_params,input_layer_size,  hidden_layer_size,  num_labels, X, y, lambda,u)
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));
% Setup some useful variables
loss = [];
m = size(X, 1);
n = input_layer_size;
hl = hidden_layer_size;
k = num_labels;
% Initial gradient parts
Theta1_grad = zeros(size(Theta1)); % hl * (n+1)
Theta2_grad = zeros(size(Theta2)); % k * (hl+1)
delta1 = zeros(size(Theta1)); % hl * (n+1)
delta2 = zeros(size(Theta2)); % k * (hl+1)
% Initial loss
J = 0;
J_old = inf;
%SIGMOID Compute sigmoid functoon
%J = SIGMOID(z) computes the sigmoid of z.
sigmoid = @(z) (1.0 ./ (1.0 + exp(-z)));
for ite = 1:1000
    u = u /1.01;
    A1 = X; % m * n
    A1 = [ones(m,1),X]; % m * (n+1)
    Z2 = A1*Theta1'; % m * hl
    A2 = sigmoid(Z2); % m * hl
    A2 = [ones(m,1),A2]; % m * (hl+1)
    Z3 = A2*Theta2'; % m * k
    A3 = sigmoid(Z3); % m * k
    H = A3; % m * k
    J = (sum(sum(-y.*log(H))) + sum(sum(-(1-y).*log(1-H))))/m + lambda*norm(Theta1(:,2:end))^2 + lambda*norm(Theta2(:,2:end))^2;
% computes the sigma
    sigma3 = A3 - y; % m * k
    g_grad_Z2 = sigmoid(Z2).*(1-sigmoid(Z2)); % m * hl
    sigma2 = sigma3*Theta2.*[ones(m,1),g_grad_Z2]; % m * (hl+1)
    sigma2 = sigma2(:,2:end); % m * hl
% computes the dalta
    for i = 1:m
        delta2 = delta2 + sigma3(i,:)'*A2(i,:); % k * (hl+1)
        delta1 = delta1 + sigma2(i,:)'*A1(i,:); % hl * (n+1)
    end
% computes the gradient
    Theta2_grad = delta2/m + lambda*Theta2/m; % k * (hl+1)
    Theta1_grad = delta1/m + lambda*Theta1/m; % hl * (n+1)
    Theta2 = Theta2 - u*Theta2_grad; % k * (hl+1)
    Theta1 = Theta1 - u*Theta1_grad; % hl * (n+1)
    loss =[loss,J];
     if J_old - J <= 10^-8 && J_old - J >= 0
        Theta_1 = Theta1;
        Theta_2 = Theta2;
        break
     end
     J_old = J;
     Theta_1 = Theta1;
     Theta_2 = Theta2;
end
