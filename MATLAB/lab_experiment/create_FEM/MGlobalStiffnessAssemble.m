clc
clear
load Mcable_element_inf;
load Mbeam_element_inf;
load Mtower_element_inf;
%载入梁单元、索单元、塔单元的单元信息；
dof_n=55*3;%n表示自由度总数，包括支座处自由度；
K=zeros(dof_n);%K表示总体刚度矩阵

%梁单元单元刚度矩阵组集
n1=28;%n表示梁单元个数;
Es=7.2e10;%Es表示梁单元弹性模量;
for i=1:1:n1;
k=MBeamElementStiffness(Es,beamI(i),beamL(i),beamA); 
K=MBeamElementAssemble(K,k,i,i+1);
end
node_n=n1+1;%node_n表示节点个数;

%左塔单元单元刚度矩阵组集
n2=12;%n表示左塔单元个数;
Ec=7.2e10;%Ec表示塔单元弹性模量;
for i=1:1:n2;
k=MBeamElementStiffness(Ec,towerI(i),towerL(i),towerA);
K=MBeamElementAssemble(K,k,node_n+i,node_n+i+1);
end
node_n=node_n+n2+1;

%右塔单元单元刚度矩阵组集
for i=1:1:n2;
k=MBeamElementStiffness(Ec,towerI(i),towerL(i),towerA);
K=MBeamElementAssemble(K,k,node_n+i,node_n+i+1);
end
node_n=node_n+n2+1;

%索单元单元刚度矩阵组集
n3=8;%n表示索单元个数
for i=1:1:n3
k=MCableElementStiffness(cableEA(i),cableL(i),cableAngle(i));
K=MCableElementAssemble(K,k,cableNode(i,1),cableNode(i,2));
end
element_n=n1+n2*2+n3;%element_n表示单元总数;

fprintf('总体刚度矩阵单元共：%d\n',element_n);
fprintf('总体刚度矩阵节点共：%d\n',node_n);
fprintf('总体刚度矩阵自由度共：%d\n',dof_n);

%进行边界条件处理
Constrained_Dof=[2,11,20,68,77,86,88,89,90,127,128,129];
m=0;
for i=1:1:length(Constrained_Dof)
SubDof=Constrained_Dof(i)-m;
K=K([1:(SubDof-1),(SubDof+1):(dof_n-m)],[1:(SubDof-1),(SubDof+1):(dof_n-m)]);
m=m+1;
end
fprintf('引入边界条件后总体刚度矩阵自由度共：%d\n',length(K));
save K K