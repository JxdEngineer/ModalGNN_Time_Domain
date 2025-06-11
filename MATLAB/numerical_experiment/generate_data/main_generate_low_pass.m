%% load data
clc
clear
load truss_200
dt = 1/200; % truss_500, 200
% dt = 1/100; % truss_1000
%% generate low-pass time-history features
freq_limit = 20; % pass frequencies below
acceleration_time_out_lowpass = cell(length(node_out),1);
for i = 1:length(node_out)
    acceleration_time_out_lowpass{i} = lowpass(acceleration_time_out{i}',freq_limit,1/dt,ImpulseResponse="iir",Steepness=0.95)';  % must use transpose!
    disp(['i=',num2str(i)])
end
acceleration_time_out = acceleration_time_out_lowpass;
save truss_200_lowpass node_out frequency_out modeshape_out acceleration_time_out element_out damping_out  % output time-history acceleration

[psd,f] = pwelch(acceleration_time_out_lowpass{1}(30,:),hamming(256),[],1024,1/dt);
figure
plot(f,psd)