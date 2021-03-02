function newpdf = Vasicekpdf(data1,data2,params)
%�إߤ@��Vasicek��pdf����mlecov�ѥX�@�ܲ��Ưx�},(����x�},������x�},�ѼƯx�})
dt=1/12;
Z=((data1-data2)-(params(1).*(params(2)-data2).*dt))./(2.*((params(3)).*dt));
newpdf=1/((2*pi).^0.5)*exp(-(Z.^2)/2);
end


