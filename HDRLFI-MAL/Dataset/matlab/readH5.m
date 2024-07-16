clc
clear all;

path = ['../TrainingData_7x7/',sprintf('%06d',14),'.h5'];
h5disp(path);

x1 = h5read(path,'/data1');
x2 = h5read(path,'/data2');
x3 = h5read(path,'/data3');
x4 = h5read(path,'/label');
x5 = h5read(path,'/ev_data');

an1 = 4; an2 = 4;
y1 = permute(squeeze(x1(:,:,:,an1,an2)),[3,2,1]);
y2 = permute(squeeze(x2(:,:,:,an1,an2)),[3,2,1]);
y3 = permute(squeeze(x3(:,:,:,an1,an2)),[3,2,1]);
y4 = permute(squeeze(x4(:,:,:,an1,an2)),[3,2,1]);
figure;
imshow(y1);
figure;
imshow(y2);
figure;
imshow(y3);
figure;
imshow(log(1+5000*y4)/log(1+5000));