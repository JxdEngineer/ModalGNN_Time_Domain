%% only consider the girder
clc
clear

dataset_N = 21;

acceleration_time_out = cell(dataset_N,1);
node_out = cell(dataset_N,1);
element_out = cell(dataset_N,1);
frequency_out = cell(dataset_N,1);
modeshape_out = cell(dataset_N,1);

% FEM modal analysis
load K
load M
order1=10;
[phis,lamda]=eigs(K,M,order1,'sm');
[lamda,sort_index]=sort(diag(lamda));
phi=zeros(length(K(:,1)),order1);
for i=1:order1
    phi(:,i)=phis(:,sort_index(i));
end
Constrained_Dof=[2,11,20,68,77,86, 88,89,90,127,128,129];
for k=1:1:length(Constrained_Dof)
    phi=insert(phi,Constrained_Dof(k));
end

% geometric information
beamL=[0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.3;0.2;0.2;0.3;0.2;0.2;0.2;0.2;0.3;0.2;0.2;0.3;0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.2];
% node information
node(1,:) = [-3,0]; % beam nodes
for i=2:length(beamL)+1
    node(i,1)=node(i-1,1)+beamL(i-1);
    node(i,2)=0;
end

% element information
element(1:length(beamL),:)=[1:length(beamL);2:length(beamL)+1]'; % beam element

for i = 1:dataset_N
    node_out{i} = node;
    element_out{i} = element;
    modeshape_out{i} = phi(2:3:length(node)*3,2:10);
    frequency_out{i} = [3.91;6.39;11.42;12.23;13.25;16.52;16.78;17.36;22.64;25.56];
end

T_len = 4000;
acc = ones(length(node),T_len+1);
% apply constraints
acc(1,:) = 0;
acc(4,:) = 0;
acc(7,:) = 0;
acc(23,:) = 0;
acc(26,:) = 0;
acc(29,:) = 0;
% load experimental data
fs = 200; % sampling frequency
for i = 1:dataset_N
    load([pwd,'\',num2str(i),'.mat'])
    [~,T1] = max(abs(signal(:,12)')); % find absolute peak to choose data segment

    if ismember(i,[4,5,6,7,10,11]) % forced vbration, skip weird peaks
        T_shift = -950;
    else
        T_shift = 0;
    end

    acc(5,:) = signal(T1-T_shift:T1+T_len-T_shift,13)';acc(5,:) = acc(5,:)-mean(acc(5,:));
    acc(9,:) = signal(T1-T_shift:T1+T_len-T_shift,12)';acc(9,:) = acc(9,:)-mean(acc(9,:));
    acc(12,:) = signal(T1-T_shift:T1+T_len-T_shift,10)';acc(12,:) = acc(12,:)-mean(acc(12,:));
    acc(15,:) = signal(T1-T_shift:T1+T_len-T_shift,9)';acc(15,:) = acc(15,:)-mean(acc(15,:));
    acc(18,:) = signal(T1-T_shift:T1+T_len-T_shift,30)';acc(18,:) = acc(18,:)-mean(acc(18,:));
    acc(21,:) = signal(T1-T_shift:T1+T_len-T_shift,27)';acc(21,:) = acc(21,:)-mean(acc(21,:));

    % % apply band-pass filter
    % acc_filtered = bandpass(acc',[2,14.5],fs,ImpulseResponse="iir",Steepness=0.95)';
    % apply low-pass filter
    acc_filtered = lowpass(acc',14.5,fs,ImpulseResponse="iir",Steepness=0.95)';
    % apply no filter
    % acc_filtered = acc;

    acceleration_time_out{i} = acc_filtered;
end

save dataset2 node_out frequency_out modeshape_out acceleration_time_out element_out
%% psd analysis
dataNo = 5;
close all
signal_acc = [acceleration_time_out{dataNo}(5,101:end);
    acceleration_time_out{dataNo}(9,101:end);
    acceleration_time_out{dataNo}(12,101:end);
    acceleration_time_out{dataNo}(15,101:end);
    acceleration_time_out{dataNo}(18,101:end);
    acceleration_time_out{dataNo}(21,101:end)];
nfft = 1024*2;
window = hamming(nfft/4);

figure
for i = 1:6
    [psd,f] = pwelch(signal_acc(i, :),window,[],nfft,fs);
    subplot(6,2,i*2-1)
    plot(signal_acc(i, :))
    ylim([-max(max(signal_acc))*1.2,max(max(signal_acc))*1.2])
    subplot(6,2,i*2-0)
    hold on
    plot(f,psd)
    xlim([0,15])
end
sgtitle(['test ',num2str(dataNo)])
%% plot signals
close all
for i = 1:21
    subplot(7,3,i)
    plot(100:T_len, acceleration_time_out{i}([5,9,12,15,18,21],101:end))
    title(['test ',num2str(i)])
end