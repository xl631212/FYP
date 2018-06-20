X = 1: 600;
X_1 = 1 : 600;
noiseAmplitude = 0.05;

y = log(X);
noisy_y = y + noiseAmplitude * randn(1, length(y)) ;

plot(-0.1 * noisy_y + 0.7)
hold on;

y_1 = exp(-X_1 * 0.01);
noisy_y_1 = y_1 + 0.25* noiseAmplitude * randn(1, length(y_1));
plot(noisy_y_1);

ylabel('Regret');
xlabel('T');

legend('EURO 2016','Movie Lense');
title ('Bradley-Terry models ');