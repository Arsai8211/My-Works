function[SjA]=SjA(sumrptm,tm,Pjtm,Ptm,r,Q)
%CDO的評價公式,計算公平利率,(支付利率矩陣,付息日矩陣,j分券剩餘本金矩陣,總剩餘本金矩陣,無風險利率矩陣,模擬次數)
UpSjA=sumrptm.*exp(-r.*tm).*Pjtm./Ptm; %將評價的公式分成分子分母最後再相除,UpSjA為分子,DownSjA為分母
Pjtm(Pjtm<=0)=1; %防止除以0
DownSjA=Pjtm.*exp(-r.*tm);
sumUpSjA=sum(UpSjA,2); %沿著Y軸加總後矩陣剩下1*1*Q維
sumDownSjA=sum(DownSjA,2);
SjA=sumUpSjA./sumDownSjA;
SjA=sum(SjA,3);%加總後除以模擬次數即得公平利率
SjA=SjA./Q;