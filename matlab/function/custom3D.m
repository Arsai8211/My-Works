function[custom3D]=custom3D(start,ended,divide,N,Q)
%將一維等差矩陣依照廠商家數N,模擬次數Q擴展至三維(起點,終點,幾等分,廠商家數,模擬次數)
threeDpre1=linspace(start,ended,divide);%建立一個以start為起點,以ended為終點,divide等分的一維等差矩陣
threeDpre2=repmat(threeDpre1,N,1);%擴展至二維
custom3D=repmat(threeDpre2,1,1,Q);%擴展至三維