function m=MCableElementMass(lineRou,l)
%lineRou��ʾ������λ��������;
%deg��ʾ��������������ϵ�Ƕ�
%l��ʾ������Ԫ����
m=lineRou*l/4*[2,0,1,0;0,2,0,1;1,0,2,0;0,1,0,2];
end

