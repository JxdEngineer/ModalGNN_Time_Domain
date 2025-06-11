%% only consider the girder
clc
clear

acceleration_time_out = cell(2,1);
node_out = cell(2,1);
element_out = cell(2,1);
frequency_out = cell(2,1);
modeshape_out = cell(2,1);

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
Constrained_Dof=[2,11,20,68+6,77+6,86+6,88+6,89+6,90+6,127+6,128+6,129+6]; 
for k=1:1:length(Constrained_Dof)
    phi=insert(phi,Constrained_Dof(k));
end

% geometric information
beamL=[0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.3;0.2;0.095;0.01;0.095;0.3;0.2;0.2;0.2;0.2;0.3;0.2;0.2;0.3;0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.2];
% node information
node(1,:) = [-3,0]; % beam nodes
for i=2:31
    node(i,1)=node(i-1,1)+beamL(i-1);
    node(i,2)=0;
end

% element information
element(1:30,:)=[1:30;2:31]'; % beam element

for i = 1:2
    node_out{i} = node;
    element_out{i} = element;
    modeshape_out{i} = phi(2:3:length(node)*3,2:10);
    frequency_out{i} = [3.68;6.26;11.42;12.23;13.25;16.52;16.78;17.36;22.64;25.56];
end

T1 = 2360;
T_len = 4000;
acc = ones(length(node),T_len+1);
% apply constraints
acc(1,:) = 0;
acc(4,:) = 0;
acc(7,:) = 0;
acc(25,:) = 0;
acc(28,:) = 0;
acc(31,:) = 0;
% load experimental data 1
load acc1
a8=signal(T1:T1+T_len,12)';
a5=signal(T1:T1+T_len,10)';
a4=signal(T1:T1+T_len,9)';
a7=signal(T1:T1+T_len,30)';
a2=signal(T1:T1+T_len,27)';
acc(9,:) = a8;
acc(14,:) = a5;
acc(17,:) = a4;
acc(20,:) = a7;
acc(23,:) = a2;
acceleration_time_out{1} = acc;
% load experimental data 2
load acc2
a8=signal(T1:T1+T_len,12)';
a5=signal(T1:T1+T_len,10)';
a4=signal(T1:T1+T_len,9)';
a7=signal(T1:T1+T_len,30)';
a2=signal(T1:T1+T_len,27)';
acc(9,:) = a8;
acc(14,:) = a5;
acc(17,:) = a4;
acc(20,:) = a7;
acc(23,:) = a2;
acceleration_time_out{2} = acc;

save dataset node_out frequency_out modeshape_out acceleration_time_out element_out
%% psd analysis
close all
deltaT = 1/100;
nfft = 1024*2;
window = hamming(nfft/4);
[psd_a2,f] = pwelch(a2,window,[],nfft,1/deltaT);
[psd_a4,f] = pwelch(a4,window,[],nfft,1/deltaT);
[psd_a5,f] = pwelch(a5,window,[],nfft,1/deltaT);
[psd_a7,f] = pwelch(a7,window,[],nfft,1/deltaT);
[psd_a8,f] = pwelch(a8,window,[],nfft,1/deltaT);
figure
hold on
plot(f,psd_a2)
plot(f,psd_a4)
plot(f,psd_a5)
plot(f,psd_a7)
plot(f,psd_a8)