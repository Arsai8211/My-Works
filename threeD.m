function[threeD]=threeD(array,T,Q)
%將一維矩陣array依照時間數T,模擬次數Q擴展成N*T*Q的三維矩陣
threeDpre=repmat(array,1,T);%矩陣擴展至二維N*T
threeD=repmat(threeDpre,1,1,Q);%擴展為三維N*T*Q