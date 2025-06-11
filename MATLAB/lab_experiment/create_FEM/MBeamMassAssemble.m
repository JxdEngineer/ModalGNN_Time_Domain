function y = MBeamMassAssemble(M,m,i,j)
%BeamAssemble   This function assembles the element mass
%               matrix k of the beam element with nodes
%               i and j into the global mass matrix M.
%               This function returns the global mass  
%               matrix M after the element mass matrix  
%               M is assembled.
M(3*i-2,3*i-2) = M(3*i-2,3*i-2) + m(1,1);
M(3*i-2,3*i-1) = M(3*i-2,3*i-1) + m(1,2);
M(3*i-2,3*i) = M(3*i-2,3*i) + m(1,3);
M(3*i-2,3*j-2) = M(3*i-2,3*j-2) + m(1,4);
M(3*i-2,3*j-1) = M(3*i-2,3*j-1) + m(1,5);
M(3*i-2,3*j) = M(3*i-2,3*j) + m(1,6);

M(3*i-1,3*i-2) = M(3*i-1,3*i-2) + m(2,1);
M(3*i-1,3*i-1) = M(3*i-1,3*i-1) + m(2,2);
M(3*i-1,3*i) = M(3*i-1,3*i) + m(2,3);
M(3*i-1,3*j-2) = M(3*i-1,3*j-2) + m(2,4);
M(3*i-1,3*j-1) = M(3*i-1,3*j-1) + m(2,5);
M(3*i-1,3*j) = M(3*i-1,3*j) + m(2,6);

M(3*i,3*i-2) = M(3*i,3*i-2) + m(3,1);
M(3*i,3*i-1) = M(3*i,3*i-1) + m(3,2);
M(3*i,3*i) = M(3*i,3*i) + m(3,3);
M(3*i,3*j-2) = M(3*i,3*j-2) + m(3,4);
M(3*i,3*j-1) = M(3*i,3*j-1) + m(3,5);
M(3*i,3*j) = M(3*i,3*j) + m(3,6);

M(3*j-2,3*i-2) = M(3*j-2,3*i-2) + m(4,1);
M(3*j-2,3*i-1) = M(3*j-2,3*i-1) + m(4,2);
M(3*j-2,3*i) = M(3*j-2,3*i) + m(4,3);
M(3*j-2,3*j-2) = M(3*j-2,3*j-2) + m(4,4);
M(3*j-2,3*j-1) = M(3*j-2,3*j-1) + m(4,5);
M(3*j-2,3*j) = M(3*j-2,3*j) + m(4,6);

M(3*j-1,3*i-2) = M(3*j-1,3*i-2) + m(5,1);
M(3*j-1,3*i-1) = M(3*j-1,3*i-1) + m(5,2);
M(3*j-1,3*i) = M(3*j-1,3*i) + m(5,3);
M(3*j-1,3*j-2) = M(3*j-1,3*j-2) + m(5,4);
M(3*j-1,3*j-1) = M(3*j-1,3*j-1) + m(5,5);
M(3*j-1,3*j) = M(3*j-1,3*j) + m(5,6);

M(3*j,3*i-2) = M(3*j,3*i-2) + m(6,1);
M(3*j,3*i-1) = M(3*j,3*i-1) + m(6,2);
M(3*j,3*i) = M(3*j,3*i) + m(6,3);
M(3*j,3*j-2) = M(3*j,3*j-2) + m(6,4);
M(3*j,3*j-1) = M(3*j,3*j-1) + m(6,5);
M(3*j,3*j) = M(3*j,3*j) + m(6,6);

y = M;