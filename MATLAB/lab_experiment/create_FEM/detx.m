function y = detx(A,m,n)
%这个函数用于提取模态矩阵中某一振型向量的整体坐标系X坐标的增量
%A表示模态矩阵某一阶振型向量
%m表示开始提取的行数(m是3的倍数+1）
%n表示终止提取的行数(n是3的倍数)
detx=zeros(n/3,1);
for k=m:1:n/3
    detx(k)=A(3*(k-1)+1);
end
y=detx;
end

