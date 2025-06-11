clear
clc
%载入梁单元、索单元、塔单元的单元信息；
load Mcable_element_inf;
load Mbeam_element_inf;
load Mtower_element_inf;
%载入梁单元、索单元、塔单元的单元信息；
dof_n=55*3;%n表示自由度总数，包括支座处自由度；
M=zeros(dof_n);%M表示总体刚度矩阵

%梁单元单元刚度矩阵组集
n1=28;%n表示梁单元个数;
for i=1:1:n1;
    m=MBeamElementMass_L(beamM(i),beamL(i),beam_m);
    M=MBeamMassAssemble(M,m,i,i+1);
end
node_n=n1+1;%node_n表示节点个数;

%左塔单元单元刚度矩阵组集
n2=12;%n表示左塔单元个数;
for i=1:1:n2;
    if i<=8
        m=MBeamElementMass_R(towerM(i),towerL(i),tower_mdown);
    else
        m=MBeamElementMass_R(towerM(i),towerL(i),tower_mup);
    end
    M=MBeamMassAssemble(M,m,node_n+i,node_n+i+1);
end
node_n=node_n+n2+1;

%右塔单元单元刚度矩阵组集
for i=1:1:n2;
    if i<=8
        m=MBeamElementMass_R(towerM(i),towerL(i),tower_mdown);
    else
        m=MBeamElementMass_R(towerM(i),towerL(i),tower_mup);
    end
    M=MBeamMassAssemble(M,m,node_n+i,node_n+i+1);
end
node_n=node_n+n2+1;

%索单元单元刚度矩阵组集
n3=8;%n表示索单元个数
for i=1:1:n3
    m=MCableElementMass(cableM(i),cableL(i));
    M=MCableMassAssemble(M,m,cableNode(i,1),cableNode(i,2));
end
element_n=n1+n2*2+n3;%element_n表示单元总数;

fprintf('总体质量矩阵单元共：%d\n',element_n);
fprintf('总体质量矩阵节点共：%d\n',node_n);
fprintf('总体质量矩阵自由度共：%d\n',dof_n);

%进行边界条件处理
Constrained_Dof=[2,11,20,68,77,86,88,89,90,127,128,129];
p=0;%p是一个标记去除质量矩阵的阶数的标记
for i=1:1:length(Constrained_Dof)
    SubDof=Constrained_Dof(i)-p;
    M=M([1:(SubDof-1),(SubDof+1):(dof_n-p)],[1:(SubDof-1),(SubDof+1):(dof_n-p)]);%将被约束处自由度的行列矩阵变为0或是删除
    p=p+1;
end
fprintf('引入边界条件后总体质量矩阵自由度共：%d\n',length(M));
save M M
