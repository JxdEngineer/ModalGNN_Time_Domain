clc
clear
load Mcable_element_inf;
load Mbeam_element_inf;
load Mtower_element_inf;
%��������Ԫ������Ԫ������Ԫ�ĵ�Ԫ��Ϣ��
dof_n=55*3;%n��ʾ���ɶ�����������֧�������ɶȣ�
K=zeros(dof_n);%K��ʾ����նȾ���

%����Ԫ��Ԫ�նȾ����鼯
n1=28;%n��ʾ����Ԫ����;
Es=7.2e10;%Es��ʾ����Ԫ����ģ��;
for i=1:1:n1;
k=MBeamElementStiffness(Es,beamI(i),beamL(i),beamA); 
K=MBeamElementAssemble(K,k,i,i+1);
end
node_n=n1+1;%node_n��ʾ�ڵ����;

%������Ԫ��Ԫ�նȾ����鼯
n2=12;%n��ʾ������Ԫ����;
Ec=7.2e10;%Ec��ʾ����Ԫ����ģ��;
for i=1:1:n2;
k=MBeamElementStiffness(Ec,towerI(i),towerL(i),towerA);
K=MBeamElementAssemble(K,k,node_n+i,node_n+i+1);
end
node_n=node_n+n2+1;

%������Ԫ��Ԫ�նȾ����鼯
for i=1:1:n2;
k=MBeamElementStiffness(Ec,towerI(i),towerL(i),towerA);
K=MBeamElementAssemble(K,k,node_n+i,node_n+i+1);
end
node_n=node_n+n2+1;

%����Ԫ��Ԫ�նȾ����鼯
n3=8;%n��ʾ����Ԫ����
for i=1:1:n3
k=MCableElementStiffness(cableEA(i),cableL(i),cableAngle(i));
K=MCableElementAssemble(K,k,cableNode(i,1),cableNode(i,2));
end
element_n=n1+n2*2+n3;%element_n��ʾ��Ԫ����;

fprintf('����նȾ���Ԫ����%d\n',element_n);
fprintf('����նȾ���ڵ㹲��%d\n',node_n);
fprintf('����նȾ������ɶȹ���%d\n',dof_n);

%���б߽���������
Constrained_Dof=[2,11,20,68,77,86,88,89,90,127,128,129];
m=0;
for i=1:1:length(Constrained_Dof)
SubDof=Constrained_Dof(i)-m;
K=K([1:(SubDof-1),(SubDof+1):(dof_n-m)],[1:(SubDof-1),(SubDof+1):(dof_n-m)]);
m=m+1;
end
fprintf('����߽�����������նȾ������ɶȹ���%d\n',length(K));
save K K