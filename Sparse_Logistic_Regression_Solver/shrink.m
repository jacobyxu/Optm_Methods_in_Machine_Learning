function [new] = shrink(x)
global lamda
global u
new = ones(1,size(x,2));
for i = 1: size(x,2)
    if x(i) > u*lamda
        new(i) = x(i) - u*lamda;
    else if x(i) > -u*lamda
            new(i) = 0;
        else
            new(i) = x(i) + u*lamda;
        end
    end
end

    