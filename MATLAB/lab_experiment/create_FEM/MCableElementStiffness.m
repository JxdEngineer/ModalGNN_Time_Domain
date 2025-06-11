function cablek=MCableElementStiffness(EA,L,deg)
%k表示拉索换算弹性模量计算刚度EA/L;
%deg表示拉索与总体坐标系角度
cablek=EA/L*[cos(deg/180*pi)^2,sin(deg/180*pi)*cos(deg/180*pi),-cos(deg/180*pi)^2,-sin(deg/180*pi)*cos(deg/180*pi);
        sin(deg/180*pi)*cos(deg/180*pi),sin(deg/180*pi)^2,-sin(deg/180*pi)*cos(deg/180*pi),-sin(deg/180*pi)^2;
        -cos(deg/180*pi)^2,-sin(deg/180*pi)*cos(deg/180*pi),cos(deg/180*pi)^2,sin(deg/180*pi)*cos(deg/180*pi);
        -sin(deg/180*pi)*cos(deg/180*pi),-sin(deg/180*pi)^2,sin(deg/180*pi)*cos(deg/180*pi),sin(deg/180*pi)^2];
end 
