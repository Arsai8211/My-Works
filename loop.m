function[Vt,Dt,sigma,mu,Zi]=loop(start,ended,company,Zi,Zim,V,D,sigmaA,mupre,Vt,Dt,sigma,mu)
%資產池內有些公司發行多個債券,因此計算損失額度時同家公司發行的債券需要使用同組參數
%(債券的起始編號,債券的截止編號,發行公司的編號,分配後隨機變數的矩陣,原始隨機變數的矩陣,原始公司資產矩陣,原始公司負債矩陣,原始資產波動度矩陣,原始mu矩陣,分配後公司資產矩陣,分配後公司負債矩陣,分配後資產波動度矩陣,分配後mu矩陣)
for i=start:ended%將隨機變數Zim,公司資產V,負債D,波動度sigmaA,mupre依照編號分配到各自的位置
    Vt(i,1)=V(company,1);
    Dt(i,1)=D(company,1);
    sigma(i,1)=sigmaA(company,1);
    mu(i,1)=mupre(company,1);
    Zpre1=Zim(company,:,:);
    Zpre2=repmat(Zpre1,ended-start+1,1,1);
end
%Zi如果為0則沒有需合併的矩陣
if Zi~=0
    Zi=[Zi;Zpre2];
else
    Zi=Zpre2;
end%輸出公司資產矩陣Vt,負債矩陣Dt,波動度矩陣sigma,mu矩陣,隨機變數矩陣Zi        