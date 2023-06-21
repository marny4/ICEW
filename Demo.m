% This matlab code implements the Infrared Maritime Target Detection Based on
% Iterative Corner and Edge Weights in Tensor Decomposition.

% Written by Enzhong Zhao 
% 2023-6-20

clc
clear all
close all

%% Adjust these for better performance
patchSize = 60; 
slideStep = 40;
lambdaL = 0.6;
mu = 0.001;
alpha = 0.2;

%% Input Image
img = imread('1.png');
if ndims( img ) == 3
    img = rgb2gray( img );
end
img = double(img); 
[m,n]=size(img);

%% constrcut patch tensor of original image
tenD = gen_patch_ten(img, patchSize, slideStep);
[n1,n2,n3] = size(tenD);

%% The proposed model
lambda = lambdaL / sqrt(max(n1,n2)*n3); 
[tenT] = ICEW(tenD,lambda,mu,alpha,patchSize, slideStep,m,n);  

%% recover the target and background image
tic
tarImg = res_patch_ten_mean(tenT, img, patchSize, slideStep);
toc
tarImg = uint8(tarImg);
maxv = max(max(double(img)));
E = uint8( mat2gray(tarImg)*maxv ); 
E = im2bw(E,0);
figure
subplot(121),imshow(img,[]),title('Origin Image');
subplot(122),imshow(E,[]),title('Output Image');

