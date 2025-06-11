load K
load M
orders=10;
[ModalShapes,frequency]=eigs(K,M,orders,'sm');
[frequency,sort_index]=sort(diag(frequency));
frequency=sqrt(frequency)/2/pi;
ModalShape=zeros(length(K(:,1)),orders);  %∑÷≈‰ø’º‰
for i=1:orders
    ModalShape(:,i)=ModalShapes(:,sort_index(i));
end
Constrained_Dof=[2,11,20,68,77,86,88,89,90,127,128,129];
for k=1:1:length(Constrained_Dof)
    ModalShape=insert(ModalShape,Constrained_Dof(k));
end
clear ModalShapes i k
save frequency frequency;
save Modalshape ModalShape;