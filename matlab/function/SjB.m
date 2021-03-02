function[SjB]=SjB(deltatm,r,tm,Pjtm,sumDl,Q)
%CDO的評價公式,計算來自資產池的利息,(付息日間隔,無風險利率矩陣,付息日矩陣,j分券剩餘本金矩陣,累積損失矩陣,模擬次數)
SjPL=deltatm.*exp(-r.*tm).*Pjtm;
sumSjPL=sum(SjPL,2);
SjB=sumDl./sumSjPL;
SjB=sum(SjB,3);%加總後除以模擬次數即得應付利率
SjB=SjB./Q;