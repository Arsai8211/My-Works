function newpdf = Vasicekpdf(data1,data2,params)
%建立一個Vasicek的pdf提供mlecov解出共變異數矩陣,(當期矩陣,落後期矩陣,參數矩陣)
dt=1/12;
Z=((data1-data2)-(params(1).*(params(2)-data2).*dt))./(2.*((params(3)).*dt));
newpdf=1/((2*pi).^0.5)*exp(-(Z.^2)/2);
end


