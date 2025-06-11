function m=MCableElementMass(lineRou,l)
%lineRou表示拉索单位长度质量;
%deg表示拉索与总体坐标系角度
%l表示拉索单元长度
m=lineRou*l/4*[2,0,1,0;0,2,0,1;1,0,2,0;0,1,0,2];
end

