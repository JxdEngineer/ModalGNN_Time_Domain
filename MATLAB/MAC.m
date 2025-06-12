
function[mac]=MAC(fi1,fi2)
fi1=real(fi1);
fi2=real(fi2);

l=length(fi1);         %fi1 and fi2 should have the same length
fi12=sum(fi1.*fi2);
fi11=sum(fi1.*fi1);
fi22=sum(fi2.*fi2);
mac=fi12^2/fi11/fi22;
