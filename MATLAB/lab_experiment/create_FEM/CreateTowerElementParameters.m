clear;clc;
a2=0.015;                  %����������Χ����
t2=0.002;                  %�������ܺ��
rou=2799.673269;          %���������ܶ�
tower_mdown=0.4;                    %�����²����ؿ�����
tower_mup=0.2;                    %�����ϲ����ؿ�����
towerA=a2^2-(a2-2*t2)^2;
towerL=[0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1;0.1];
Itower=a2^4/12-(a2-2*t2)^4/12;
towerI=[Itower;Itower;Itower;Itower;Itower;Itower;Itower;Itower;Itower;Itower;Itower;Itower];
for k=1:1:length(towerL)
    if k<=(length(towerL)-1)
        towerM(k,1)=rou*towerA;
        else
        towerM(k,1)=rou*towerA*1;
    end
end
towerLeftNode(1,1)=1.2;
towerLeftNode(1,2)=-0.2;
for k=1:1:length(towerL)
    towerLeftNode(k+1,2)=towerLeftNode(k,2)+towerL(k);
    towerLeftNode(k+1,1)=1.2;
end
towerRightNode(1,1)=4.8;
towerRightNode(1,2)=-0.2;
for k=1:1:length(towerL)
    towerRightNode(k+1,2)=towerRightNode(k,2)+towerL(k);
    towerRightNode(k+1,1)=4.8;
end
save Mtower_element_inf tower*
%��������Ԫ
