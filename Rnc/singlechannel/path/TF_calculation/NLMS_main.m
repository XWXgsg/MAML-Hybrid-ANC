close all;clc;clear;
load test_0414New_Speaker_FL_4person.mat
fs_ori = 1/Signal_1.x_values.increment;
fs_ori1 = 1/Signal_0.x_values.increment;
fs = 2000;
sensitivity = 0.058306385;
input = resample(Signal_1.y_values.values([673351:929351 980551:1236551],1),fs,fs_ori);
output = sensitivity*resample(Signal_0.y_values.values([673351:929351 980551:1236551],1),fs,fs_ori1);
un = input;
dn = output;
M = 128; 
mu = ;
leak = 0;
S = NLMSinit(zeros(M,1),mu,leak);
[yn,en,S,W] = NLMSadapt(un,dn,S);
plot(W);
u = un(128:255,1);
sum = u'*u;