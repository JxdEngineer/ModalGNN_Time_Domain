function y = MBeamElementMass_R(lineRou,L,m)
%BeamElementMass   This function returns the element 
%                       mass matrix for a beam   
%                       element with distributed mass Rou
%                       and length L.
%                       The size of the element stiffness 
%                       matrix is 6 x 6.
%这个梁单元右端有配重
%  lineRou是单元线密度，L是单元长度,m是节点配重质量；
A= [0,0,0,0,0,0;
    0,0,0,0,0,0;
    0,0,0,0,0,0;
    0,0,0,m,0,0;
    0,0,0,0,m,0;
    0,0,0,0,0,0];
B=lineRou*L/420*[140,0,0,70,0,0;
    0,156,22*L,0,54,-13*L;
    0,22*L,4*L^2,0,13*L,-3*L^2;
    70,0,0,140,0,0;
    0,54,13*L,0,156,-22*L;
    0,-13*L,-3*L^2,0,-22*L,4*L^2];
y=A+B;