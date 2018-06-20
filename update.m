function [z ,n] = update(x,n,alpha,w)
for i = 1: size(x)
    g = (p - y) * x(i);
    s = (sqrt(n(i) + g*g) - sqrt(n(i))) / alpha ;
    z(i) = z(i) + (g - s * w(i));
    n(i) = n(i) + g*g
end

end