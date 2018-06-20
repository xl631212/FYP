function loss = log_loss(y, p)

%%% compute the log loss of  probability p given
%    a true target y.
p = max(min(p, 1. - 10e-15), 10e-15);

if y == 1 
    loss = -log(p); 
else
    loss = -log(1. -p);
end