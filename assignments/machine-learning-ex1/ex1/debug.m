clear ; close all; clc

fprintf('Debug Feature Normalize ...\n')

A = [1 2; 3 4];

[A mu sigma] = featureNormalize(A);

fprintf('Means: [%f %f] \n', mu);
fprintf('Standard error: [%f %f] \n', sigma);
fprintf('[%f %f %f %f] \n', A(:)');

[A mu sigma] = featureNormalize(A);

fprintf('Means: [%f %f] \n', mu);
fprintf('Standard error: [%f %f] \n', sigma);
