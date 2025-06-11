function y=insert(A,n)
%这是一个为矩阵插入指定行(value=0)的函数
%   A表示待插入的矩阵，n表示要插入的行数。
if n == 1  % the case of inserting the first row
    M = [zeros(1,length(A(1,:)));A];
elseif n == length(A(:,1))+1  % the case of inserting the last row
    M = [A;zeros(1,length(A(1,:)))];
else
    for k=1:1:n-1
        M(k,:)=A(k,:);
    end
    for k=n+1:1:(size(A,1)+1)
        M(k,:)=A(k-1,:);
    end
end
y=M;
end