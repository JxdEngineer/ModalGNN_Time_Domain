function y=insert(A,n)
%����һ��Ϊ�������ָ����(value=0)�ĺ���
%   A��ʾ������ľ���n��ʾҪ�����������
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