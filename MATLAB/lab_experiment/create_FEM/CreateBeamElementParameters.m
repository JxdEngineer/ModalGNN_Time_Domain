clear;clc;
a1=0.015;                     %��������Χ����
t1=0.0012;                    %�����ܱں�
I=a1^4/12-(a1-2*t1)^4/12;
rous=2799.673269;             %���Ͻ��ܶ�
beam_m=0.5;                       %�����ؿ�����
beamL=[0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.3;0.2;0.2;0.3;0.2;0.2;0.2;0.2;0.3;0.2;0.2;0.3;0.2;0.2;0.2;0.2;0.2;0.2;0.2;0.2];
beamI=[I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I;I]; %������15��15��1.2mm
beamA=a1^2-(a1-2*t1)^2;
beamM=zeros(length(beamL),1);
p=1;
for k=1:1:length(beamL)
    beamM(k)=rous*beamA;
end
beamNode(1,1)=0;              %���������ڵ�����
beamNode(1,2)=0;
for k=1:1:length(beamL)
    beamNode(k+1,1)=beamNode(k,1)+beamL(k);
    beamNode(k+1,2)=0;
end
save Mbeam_element_inf b*
%��������Ԫ
