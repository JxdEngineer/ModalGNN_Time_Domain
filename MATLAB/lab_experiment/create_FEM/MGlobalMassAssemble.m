clear
clc
%��������Ԫ������Ԫ������Ԫ�ĵ�Ԫ��Ϣ��
load Mcable_element_inf;
load Mbeam_element_inf;
load Mtower_element_inf;
%��������Ԫ������Ԫ������Ԫ�ĵ�Ԫ��Ϣ��
dof_n=55*3;%n��ʾ���ɶ�����������֧�������ɶȣ�
M=zeros(dof_n);%M��ʾ����նȾ���

%����Ԫ��Ԫ�նȾ����鼯
n1=28;%n��ʾ����Ԫ����;
for i=1:1:n1;
    m=MBeamElementMass_L(beamM(i),beamL(i),beam_m);
    M=MBeamMassAssemble(M,m,i,i+1);
end
node_n=n1+1;%node_n��ʾ�ڵ����;

%������Ԫ��Ԫ�նȾ����鼯
n2=12;%n��ʾ������Ԫ����;
for i=1:1:n2;
    if i<=8
        m=MBeamElementMass_R(towerM(i),towerL(i),tower_mdown);
    else
        m=MBeamElementMass_R(towerM(i),towerL(i),tower_mup);
    end
    M=MBeamMassAssemble(M,m,node_n+i,node_n+i+1);
end
node_n=node_n+n2+1;

%������Ԫ��Ԫ�նȾ����鼯
for i=1:1:n2;
    if i<=8
        m=MBeamElementMass_R(towerM(i),towerL(i),tower_mdown);
    else
        m=MBeamElementMass_R(towerM(i),towerL(i),tower_mup);
    end
    M=MBeamMassAssemble(M,m,node_n+i,node_n+i+1);
end
node_n=node_n+n2+1;

%����Ԫ��Ԫ�նȾ����鼯
n3=8;%n��ʾ����Ԫ����
for i=1:1:n3
    m=MCableElementMass(cableM(i),cableL(i));
    M=MCableMassAssemble(M,m,cableNode(i,1),cableNode(i,2));
end
element_n=n1+n2*2+n3;%element_n��ʾ��Ԫ����;

fprintf('������������Ԫ����%d\n',element_n);
fprintf('������������ڵ㹲��%d\n',node_n);
fprintf('���������������ɶȹ���%d\n',dof_n);

%���б߽���������
Constrained_Dof=[2,11,20,68,77,86,88,89,90,127,128,129];
p=0;%p��һ�����ȥ����������Ľ����ı��
for i=1:1:length(Constrained_Dof)
    SubDof=Constrained_Dof(i)-p;
    M=M([1:(SubDof-1),(SubDof+1):(dof_n-p)],[1:(SubDof-1),(SubDof+1):(dof_n-p)]);%����Լ�������ɶȵ����о����Ϊ0����ɾ��
    p=p+1;
end
fprintf('����߽����������������������ɶȹ���%d\n',length(M));
save M M
