function cablek=MCableElementStiffness(EA,L,deg)
%k��ʾ�������㵯��ģ������ն�EA/L;
%deg��ʾ��������������ϵ�Ƕ�
cablek=EA/L*[cos(deg/180*pi)^2,sin(deg/180*pi)*cos(deg/180*pi),-cos(deg/180*pi)^2,-sin(deg/180*pi)*cos(deg/180*pi);
        sin(deg/180*pi)*cos(deg/180*pi),sin(deg/180*pi)^2,-sin(deg/180*pi)*cos(deg/180*pi),-sin(deg/180*pi)^2;
        -cos(deg/180*pi)^2,-sin(deg/180*pi)*cos(deg/180*pi),cos(deg/180*pi)^2,sin(deg/180*pi)*cos(deg/180*pi);
        -sin(deg/180*pi)*cos(deg/180*pi),-sin(deg/180*pi)^2,sin(deg/180*pi)*cos(deg/180*pi),sin(deg/180*pi)^2];
end 
