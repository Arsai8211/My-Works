function[rt]=interest(rstart,a,mur,sigmar,dW,deltatm,Time,T,Q)
%利用參數估計值模擬Q條利率的路徑(利率起始值,a,mu,sigma,winner process,付息日間隔,模擬時間長度,模擬次數)
dt=linspace(deltatm,Time,T);%付息日的累加矩陣
dt=repmat(dt,1,1,Q);%擴展為1*T*Q矩陣
dWt=cumsum(dW,2);%累加winner process
drt=a.*(mur-rstart).*dt+sigmar.*dWt;%代入Vasicek公式
rt=rstart+drt;%得到每期利率模擬值