% Follow the leader algorithm test with database 
clear all;
close all;
load spam_inst.mat
load spam_label.mat
syms w t
T = 100;
y = spam_label;
x = spam_inst;
%w = zeros(size(x));
%ft(w) = max{0, 1 - yt.* w'.*xt};
%%
% loss function: f(w) = gt*w where gt = y
for s = 1: T
       if y(s) < 0
           w_sol(s) = -1;
       else
           w_sol(s) = 1;
       end
 fun = @(w) w*sum(y(1:T)) ;
 options = optimset('Display','iter');
 [w_min,fval] = fminbnd(fun,-1,1,options);
 Regert(s) = sum(w_sol*y(1:s)) - fval;
end
figure;
plot(Regert);
hold on
  %%
% % svm loss
% % ft(w) = max{0, 1 - yt.* w'.*xt};
% % square loss = (y - f(x))^2
% L = 0;
% for s = 2 :T
%     % w_t = argmin sum(loss_function);
% %     L = symsum( max(0, 1 - y(t,:).* w'.*x(t)), t, [1 s - 1])
%     for i = 1: s - 1
%         if (1 - y(i,:).* w'.*x(i)) > 0
%             L = L + (1 - y(i,:).* w'.*x(i));
%         else
%             L = L + 0;
%         end
%     end
%     eqn = diff(L, w) == 0;
%     w_sol_svm(s-1) = solve(eqn,w);    
% end
% 
% fun = @(w) w*sum(y(1:T)) ;
%  [w_min,fval] = fminbnd(fun,-1,1);
%  Regert(s) = sum(w_sol*y(1:s)) - fval;

 %% convex quadratic function 
 % f = 15 + 0.5 * w.*w;
 for s = 2: T 
     L = symsum( 15 + 0.5 * w.*w , t, [1 s - 1]);
     eqn = diff(L, w) == 0;
     w_sol_convex(s-1) = solve(eqn,w);
     fun = @(k)  15 + 0.5 * k.*k;
     [k_min,fval] = fminbnd(fun,-1,1);
     Regert_convex(s) = 15 + 0.5 * 0.*0 - fval;
 end
 
 plot(Regert_convex);
hold on
  %% convex quadratic function 
  % ft(w) = 1/2 (w.*w' - z.*z')
 z = rand(100,1);
 
 for j = 2 : 100
     w_t = (1/(j-1)) * sum(z(1:j-1));   
     w_t_1 = (1/(j)) * sum(z(1:j));
     loss(j - 1) = w_t_1 - w_t;
     %Regert_convex_best(j-1) = (1/2)* sum(w_t*w_t - z(1:j-1)'*z(1:j-1) - w_t_1*w_t_1 + z(1:j)'*z(1:j)); 
     upper_bound(j-1) = 4*(log(j)+1);
 end
 %plot(Regert_convex_best)
 %hold on;
 plot(upper_bound)
 hold on
 %% diferent loss functions 

 %%
 syms w1;
 %f1 = 2 + w1.^3;
 fplot(@(x1) 2 + x1.^3); hold on;
 circle(-0.855768,-0.0428181,2.85709);
 hold off;
  for s = 2: T 
     L = symsum( 2 + w1*w1*w1 , t, [1 s - 1]);
     eqn = diff(L, w1) == 0;
     w_sol_1(s-1) = solve(eqn,w1);
     fun = @(k)  2 + k .* k.*k;
     [k_min,fval] = fminbnd(fun,-1,1);
     %Regert_convex(s) = 2* (s-1)* sum( ) - fval;
  end
 
 %%
 w2 = 0;
 f2 = sin(w2);
 fplot(@(x) sin(x)); hold on;
 x0 = pi/2;
 circle(x0 + (cos(x0).^2 + 1)*cot(x0), -2*cos(x0)*cot(x0),(cos(x0).^2 +1).^(3/2)/abs(sin(x0)));
 hold on;
 x1 = -pi/2;
 circle(x1 + (cos(x1).^2 + 1)*cot(x1), -2*cos(x1)*cot(x1),(cos(x1).^2 +1).^(3/2)/abs(sin(x1)));
 hold off;
 
 Regert = 1; 
 
 %%
 w3 = 0;
 f3 = -exp(-w3.^2);
 fplot(@(x3) -exp(-x3.^2)); hold on;
 circle(0,-1/2, 1/2);
 hold off; 
 regert = 0;

 %% stochastic data 
 R = normrnd(0,1,2500,1);
 L = 0.1;
 e = 1;
 for i = 1: 2500
     if R(i)*R(i) <= 1
         f(i) = R(i);
     else
         f(i) = R(i) + L *e ;
     end
 end
 
 for i = 1:2500
     regert(i) = log(i);
 end
 
figure; 
i = (1:2500);
hold on
plot(sqrt(i),regert)
hold on
plot(log(i),regert)

%%
clear ;
close all;
x = [1:50].';
 y = [4554 3014 2171 1891 1593 1532 1416 1326 1297 1266 ...
 	1248 1052 951 936 918 797 743 665 662 652 ...
 	629 609 596 590 582 547 486 471 462 435 ...
 	424 403 400 386 386 384 384 383 370 365 ...
 	360 358 354 347 320 319 318 311 307 290 ].';
% y = sqrt((x.^2 -1).^2) + 1;
% y = -exp(-x.^2);
m = length(y); % store the number of training examples
x = [ ones(m,1) x]; % Add a column of ones to x
n = size(x,2); % number of features
theta_vec = [0 0]';
alpha = 0.000002;
err = [0 0]';
for kk = 1:10000
	h_theta = (x*theta_vec);
	h_theta_v = h_theta*ones(1,n);
	y_v = y*ones(1,n);
	theta_vec = theta_vec - alpha*1/m*sum((h_theta_v - y_v).*x).';
	err(:,kk) = 1/m*sum((h_theta_v - y_v).*x).';
end

figure;
plot(x(:,2),y,'bs-');
hold on
plot(x(:,2),x*theta_vec,'rp-');
legend('measured', 'predicted');
grid on;
xlabel('Page index, x');
ylabel('Page views, y');
title('Measured and predicted page views');

j_theta = zeros(250, 250);   % initialize j_theta
theta0_vals = linspace(-5000, 5000, 250);
theta1_vals = linspace(-200, 200, 250);
for i = 1:length(theta0_vals)
	  for j = 1:length(theta1_vals)
		theta_val_vec = [theta0_vals(i) theta1_vals(j)]';
		h_theta = (x*theta_val_vec);
		j_theta(i,j) = 1/(2*m)*sum((h_theta - y).^2);
    end
end
figure;
surf(theta0_vals, theta1_vals,10*log10(j_theta.'));
xlabel('theta_0'); ylabel('theta_1');zlabel('10*log10(Jtheta)');
title('loss function(theta)');
figure;
contour(theta0_vals,theta1_vals,10*log10(j_theta.'))
xlabel('theta_0'); ylabel('theta_1')
title('loss function (theta)');
 
 


%%

clear ;
close all;
x = [1:50].';
% y = [4554 3014 2171 1891 1593 1532 1416 1326 1297 1266 ...
% 	1248 1052 951 936 918 797 743 665 662 652 ...
% 	629 609 596 590 582 547 486 471 462 435 ...
% 	424 403 400 386 386 384 384 383 370 365 ...
% 	360 358 354 347 320 319 318 311 307 290 ].';
%
% y = 3*x;
y = -exp(-x.^2);

m = length(y); % store the number of training examples
x = [ ones(m,1) x]; % Add a column of ones to x
n = size(x,2); % number of features
theta_batch_vec = [0 0]';
theta_stoch_vec = [0 0]';
alpha = 0.002;
err = [0 0]';
theta_batch_vec_v = zeros(10000,2);
theta_stoch_vec_v = zeros(50*10000,2);
for kk = 1:10000
	% batch gradient descent - loop over all training set
	h_theta_batch = (x*theta_batch_vec);
	h_theta_batch_v = h_theta_batch*ones(1,n);
	y_v = y*ones(1,n);
	theta_batch_vec = theta_batch_vec - alpha*1/m*sum((h_theta_batch_v - y_v).*x).';
	theta_batch_vec_v(kk,:) = theta_batch_vec;
	%j_theta_batch(kk) = 1/(2*m)*sum((h_theta_batch - y).^2);

	% stochastic gradient descent - loop over one training set at a time
	for (jj = 1:50)
		h_theta_stoch = (x(jj,:)*theta_stoch_vec);
		h_theta_stoch_v = h_theta_stoch*ones(1,n);
		y_v = y(jj,:)*ones(1,n);
		theta_stoch_vec = theta_stoch_vec - alpha*1/m*((h_theta_stoch_v - y_v).*x(jj,:)).';
		%j_theta_stoch(kk,jj) = 1/(2*m)*sum((h_theta_stoch - y).^2);
		theta_stoch_vec_v(50*(kk-1)+jj,:) = theta_stoch_vec;
	end
end

figure;
plot(x(:,2),y,'bs-');
hold on
plot(x(:,2),x*theta_batch_vec,'md-');
plot(x(:,2),x*theta_stoch_vec,'rp-');
legend('measured', 'predicted-batch','predicted-stochastic');
grid on;
xlabel('Page index, x');
ylabel('Page views, y');
title('Measured and predicted page views');

j_theta = zeros(250, 250);   % initialize j_theta
theta0_vals = linspace(-2500, 2500, 250);
theta1_vals = linspace(-50, 50, 250);
for i = 1:length(theta0_vals)
	  for j = 1:length(theta1_vals)
		theta_val_vec = [theta0_vals(i) theta1_vals(j)]';
		h_theta = (x*theta_val_vec);
		j_theta(i,j) = 1/(2*m)*sum((h_theta - y).^2);
    end
end
figure;
contour(theta0_vals,theta1_vals,10*log10(j_theta.'))
xlabel('theta_0'); ylabel('theta_1')
title('loss function(theta)');
hold on;
plot(theta_stoch_vec_v(:,1),theta_stoch_vec_v(:,2));
plot(theta_batch_vec_v(:,1),theta_batch_vec_v(:,2));
 
 

%% 
 
 
 
 
 