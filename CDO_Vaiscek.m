%Vasicek參數設定
P=100;%#Vasicek矩陣切割數
%Ronn and Verma參數設定
M=19;%廠商家數
TimeS=1;%估計Ronn and Verma使用的時間參數
S=5000;%分割數
%Pricing參數設定
Q=10000;%模擬次數
N=37;%債券個數
T=27;%時間切割段數
Time=6.75;%模擬時間長數
tm1=custom3D(Time/T,Time,T,N,Q);%付息日矩陣
deltatm=Time/T;%每段時間間隔
%Vasicek data,匯入資料料後進行AR(1)回歸作為參數起始值後,估計Vasicek模型的參數
LIBORrt=inputdata("D:/paper/vasicek/6ML 5y.xls",1,"value");%匯入利率資料(路徑,工作表,標題名稱)
[Lalpha,Lbeta,Lmse,Lk]=ARone("D:/paper/vasicek/6ML 5y.xls");%AR(1)回歸後得到alpha,beta,mse
[La,Lmur,Lsigmar]=Vasicek(Lalpha,Lbeta,Lmse,P,Lk,LIBORrt);%以AR(1)的參數作為參數估計起始值估計Vasicek的a,mu,sigma
CMSrt=inputdata("D:/paper/vasicek/5CMS-2CMS 5y.xlsx",1,"value");
[CMSalpha,CMSbeta,CMSmse,CMSk]=ARone("D:/paper/vasicek/5CMS-2CMS 5y.xlsx");
[CMSa,CMSmur,CMSsigmar]=Vasicek(CMSalpha,CMSbeta,CMSmse,P,CMSk,CMSrt);
CPrt=inputdata("D:/paper/vasicek/90CP 5y.xlsx",1,"value");
[CPalpha,CPbeta,CPmse,CPk]=ARone("D:/paper/vasicek/90CP 5y.xlsx");
[CPa,CPmur,CPsigmar]=Vasicek(CPalpha,CPbeta,CPmse,P,CPk,CPrt);
CP180rt=inputdata("D:/paper/vasicek/180CP 5y.xlsx",1,"value");
[CP180alpha,CP180beta,CP180mse,CP180k]=ARone("D:/paper/vasicek/180CP 5y.xlsx");
[CP180a,CP180mur,CP180sigmar]=Vasicek(CP180alpha,CP180beta,CP180mse,P,CP180k,CP180rt);
rrt=inputdata("D:/paper/vasicek/郵局1個月定存利率 5y.xlsx",1,"value");
[ralpha,rbeta,rmse,rk]=ARone("D:/paper/vasicek/郵局1個月定存利率 5y.xlsx");
[ra,rmur,rsigmar]=Vasicek(ralpha,rbeta,rmse,P,rk,rrt);
%將利率及模擬公司資產的隨機變數先進行模擬
Z=normrnd(0,deltatm,M+6,T,Q);%Z~N(0,deltatm)
%將隨機變數的矩陣分成模擬公司資產用的與模擬利率用的
Zim=Z(1:M,:,:);%公司特有因子
Zm=Z(M+1,:,:);%市場系統因子
Zm=repmat(Zm,N,1,1);%將市場系統因子的矩陣擴展為N*T*Q
%將利率的隨機變數從矩陣取出並模擬利率的變化
LIBORdW=Z(M+2,:,:);%從矩陣取出
LIBORM=interest(LIBORrt(Lk,1),La,Lmur,Lsigmar,LIBORdW,deltatm,Time,T,Q);%模擬利率變化
CPdW=Z(M+3,:,:);
CPM=interest(CPrt(CPk,1),CPa,CPmur,CPsigmar,CPdW,deltatm,Time,T,Q);
CP180dW=Z(M+4,:,:);
CP180M=interest(CP180rt(CP180k,1),CP180a,CP180mur,CP180sigmar,CP180dW,deltatm,Time,T,Q);
CMSdW=Z(M+5,:,:);
CMSM=interest(CMSrt(CMSk,1),CMSa,CMSmur,CMSsigmar,CMSdW,deltatm,Time,T,Q);
rdW=Z(M+6,:,:);
rpre=interest(rrt(rk,1),ra,rmur,rsigmar,rdW,deltatm,Time,T,Q);
%模擬出的負利率使其等於0
LIBORM(LIBORM<0)=0;
CPM(CPM<0)=0;
CP180M(CP180M<0)=0;
CMSM(CMSM<0)=0;
rpre(rpre<0)=0;
r=cumsum(rpre,1);%將無風險利率累加作為每一期的折現率
%匯入公司資訊與資產池內債券資訊
%依序為負債,權益波動度,權益價值,mu,本金,各債券之固定利率
Dpre=inputdata("D:/python/新2005年玉山資產池概要.xlsm","程式用","負債");
sigmaEpre=inputdata("D:/python/新2005年玉山資產池概要.xlsm","程式用","權益波動度");
VEpre=inputdata("D:/python/新2005年玉山資產池概要.xlsm","程式用","權益價值");
mupre=inputdata("D:/python/新2005年玉山資產池概要.xlsm","程式用","ROA稅後息前");
Bi=inputdata("D:/python/新2005年玉山資產池概要.xlsm","程式用2","本金");
irate=inputdata("D:/python/新2005年玉山資產池概要.xlsm","程式用2","基礎利率");
rx=rrt(rk,1);%Ronn and Verma的折現率
%預先建立一個公司資產價值與資產波動度的M*1的0矩陣以便計算後填入
V=zeros(M,1);
sigmaA=zeros(M,1);
%將權益價值VE及權益波動度sigmaE逐筆代入
for i=1:M
    VE=VEpre(i,1);
    sigmaE=sigmaEpre(i,1);
    D=Dpre(i,1);
    %VA為1倍(VE+D)至10倍(VE+D)依X軸排序
    VApre1=linspace(1*(VE+D),10*(VE+D),S);
    VApre2=repmat(VApre1,S,1);
    %sigmaA為0.01倍sigmaE至10倍sigmaE依Y軸排序
    sigmaApre1=linspace(0.01*sigmaE,10*sigmaE,S);
    sigmaAarray=repmat(sigmaApre1',1,S);
    %代入公式
    d1pre=log(VApre2./D);
    d1=(d1pre+(rx+(sigmaAarray^2)/2).*TimeS)./(sigmaAarray.*(TimeS^0.5));
    d2=d1-sigmaAarray*(TimeS^0.5);
    Nd1=normcdf(d1);
    Nd2=normcdf(d2);
    %解Ronn and Verma的聯立方程式
    apre=VApre2.*Nd1-D.*exp(-rx.*TimeS).*Nd2-VE;
    bpre=VApre2./VE.*Nd1.*sigmaAarray-sigmaE;
    a=apre./VE;%利用誤差百分比的概念
    b=bpre./sigmaE;
    c=abs(a)+abs(b);
    where=min(min(c));%找出最小誤差百分比的參數座標位置
    [row,column]=find(c==where);
    V(i,1)=VApre2(row,column);
    sigmaA(i,1)=sigmaAarray(row,column);
end
%Pricing data,先建立各參數的0矩陣
Vt=zeros(N,1);
Dt=zeros(N,1);
sigma=zeros(N,1);
mu=zeros(N,1);
rho=zeros(N,1);
%將公司的參數分配至期發行的債券
Zi=0;
[Vt,Dt,sigma,mu,Zi]=loop(1,3,1,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(4,5,2,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(6,9,3,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(10,12,4,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(13,14,5,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(15,15,6,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(16,17,7,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(18,19,8,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(20,20,9,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(21,22,10,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(23,24,11,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(25,26,12,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(27,27,13,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(28,30,14,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(31,32,15,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(33,33,16,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(34,34,17,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(35,35,18,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(36,36,2,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
[Vt,Dt,sigma,mu,Zi]=loop(37,37,19,Zi,Zim,V,Dpre,sigmaA,mupre,Vt,Dt,sigma,mu);
%將參數的矩陣擴展成三維
Bi=threeD(Bi,T,Q);
rho=threeD(rho,T,Q);
mu=threeD(mu,T,Q);
sigma=threeD(sigma,T,Q);
Dt=threeD(Dt,T,Q);
Vt=threeD(Vt,T,Q);
%債券的利率為固定利率+上一個浮動利率,此處先將固定利率放入矩陣
rConstant=irate(1,:,:);
rLIBOR=irate(2:34,:,:);
rCP=irate(35,:,:);
rCMS=irate(36,:,:);
rMax=irate(37,:,:);
%固定利率矩陣擴展成三維
rConstant=threeD(rConstant,T,Q);
rLIBOR=threeD(rLIBOR,T,Q);
rCP=threeD(rCP,T,Q);
rCMS=threeD(rCMS,T,Q);
rMax=threeD(rMax,T,Q);
%利用固定利率矩陣與模擬的利率矩陣計算各債券應支付的利率
LIBORt=rLIBOR-LIBORM;
CPt=rCP+CPM;
CMSt=rCMS+CMSM;
Max1=LIBORM-CP180M;%債券為MAX{LIBOR,180CP},因此需判斷LIBOR與180CP哪個利率較高
Max1(Max1>0)=1;
Max1(Max1<0)=0;
Max2=1-Max1;
Max3=LIBORM.*Max1+CP180M.*Max2;
Maxt=rMax+Max3;
%將利率矩陣拼接成N*T*Q
two=[rConstant;LIBORt];
three=[two;CPt];
four=[three;CMSt];
five=[four;Maxt];
five(five<0)=0;
%Pricing計算
dz=((1-(rho.^2)).^0.5).*Zi+rho.*Zm;
expopre=exp((mu-1/2.*(sigma.^2)).*deltatm+sigma.*dz);%建立exp的矩陣
expo=cumprod(expopre,2);%建立累乘exp矩陣(dim=1以y軸累乘,=2以x軸累乘)
Vtex=Vt.*expo;%得到各期模擬的公司資產
decide=Vtex-Dt;%公司價值減負債的值判斷是否違約
decide(decide>0)=1;%若大於0設為1
decide(decide<0)=0;%小於0設為0
decide=cumsum(decide,2);%以X軸累加
deduction=custom3D(0,T-1,T,N,Q);%建立一個0到T每次增加1的矩陣，兩式相減若小於0即可知道公司何時違約何時存活
survival=decide-deduction;
survival(survival<0)=0;%讓生存矩陣小於0的部分等於0
rptm=Bi.*survival.*five;%生存矩陣乘以本金和利率可得每期利息
sumrptm=sum(rptm,1);%加總利息
defaultpre=(1-survival);%違約矩陣
Ri=Vtex./Dt;%違約回復率=違約當期的資產/負債
default=cumsum(defaultpre,2);%判斷在哪一期有損失
default(default>1)=0;%有損失的那期為1其他為0
Ltm=Bi.*(1-Ri).*default;%得到損失金額
sumLtmpre=sum(Ltm,1);%將所有債券的損失加總,變成T*Q的矩陣
sumLtm=cumsum(sumLtmpre,1);%累加損失矩陣,得到各期總損失金額
Dlpre=default.*Bi.*(1-Ri).*exp(-r.*tm1);%損失現值
Dl=sum(Dlpre,1);%將所有債券的損失現值加總,變成T*Q的矩陣
sumDl=sum(Dl,2);%將T期的損失現值加總,用以計算Sj1
tm2=tm1(1,:,:);%從付息日矩陣取出T*Q的矩陣
principal=inputdata("D:/python/新2005年玉山資產池概要.xlsm","分券金額","發行金額");%匯入分券金額data
%pA至pD是A.B.C.D各分券的分券上限金額
pA=principal(4,1)+principal(3,1)+principal(2,1)+principal(1,1);
pB=principal(4,1)+principal(3,1)+principal(2,1);
pC=principal(4,1)+principal(3,1);
pD=principal(4,1);
%計算每期分券剩餘本金
P1tm=Pool(pA,pB,sumLtm);
P2tm=Pool(pB,pC,sumLtm);
P3tm=Pool(pC,pD,sumLtm);
P4tm=Pool(pD,0,sumLtm);
Ptm=P1tm+P2tm+P3tm+P4tm;%加總分券本金
%計算CDO應支付的利率
S1A=SjA(sumrptm,tm2,P1tm,Ptm,r,Q);
S2A=SjA(sumrptm,tm2,P2tm,Ptm,r,Q);
S3A=SjA(sumrptm,tm2,P3tm,Ptm,r,Q);
S4A=SjA(sumrptm,tm2,P4tm,Ptm,r,Q);
S1B=SjB(deltatm,r,tm2,P1tm,sumDl,Q);
S2B=SjB(deltatm,r,tm2,P2tm,sumDl,Q);
S3B=SjB(deltatm,r,tm2,P3tm,sumDl,Q);
S4B=SjB(deltatm,r,tm2,P4tm,sumDl,Q);
TS1=S1A+S1B;
TS2=S2A+S2B;
TS3=S3A+S3B;
TS4=S4A+S4B;
%計算最大概似法的有效性
RFparameter=[ra,rmur,rsigmar];%先將估計出來的參數排序
Lparameter=[La,Lmur,Lsigmar];
CPparameter=[CPa,CPmur,CPsigmar];
CP180parameter=[CP180a,CP180mur,CP180sigmar];
CMSparameter=[CMSa,CMSmur,CMSsigmar];
Lcov=mlecovARone(Lparameter,LIBORrt,@Vasicekpdf);%輸入參數,利率樣本,pdf得到共變異數矩陣
CPcov=mlecovARone(CPparameter,CPrt,@Vasicekpdf);
CP180cov=mlecovARone(CP180parameter,CP180rt,@Vasicekpdf);
CMScov=mlecovARone(CMSparameter,CMSrt,@Vasicekpdf);
Rcov=mlecovARone(RFparameter,rrt,@Vasicekpdf);
%把要輸出的資料和標題排序好
LcovM=["","LIBOR variance-covariance Matrix","";Lcov];
CPcovM=["","CP variance-covariance Matrix","";CPcov];
CP180covM=["","CP180 variance-covariance Matrix","";CP180cov];
CMScovM=["","CMS variance-covariance Matrix","";CMScov];
RcovM=["","Rf variance-covariance Matrix","";Rcov];
ARmatrix=["AR","alpha","Beta","MSE";"LIBOR",Lalpha,Lbeta,Lmse;"5CMS-2CMS",CMSalpha,CMSbeta,CMSmse;"90CP",CPalpha,CPbeta,CPmse;"180CP",CP180alpha,CP180beta,CP180mse;"無風險利率",ralpha,rbeta,rmse];
VSmatrix=["Vasicek","a","mu","sigma";"LIBOR",La,Lmur,Lsigmar;"5CMS-2CMS",CMSa,CMSmur,CMSsigmar;"90CP",CPa,CPmur,CPsigmar;"180CP",CP180a,CP180mur,CP180sigmar;"無風險利率",ra,rmur,rsigmar];
Spreadmatrix=["來自資產池的利率","以損失計算的公平利率","總利率";"S10","S11","TS1";S1A,S1B,TS1;"S20","S21","TS2";S2A,S2B,TS2;"S30","S31","TS3";S3A,S3B,TS3;"S40","S41","TS4";S4A,S4B,TS4];
TitleM=['公司估計價值',"公司負債價值","公司權益價值","公司總波動度","公司權益價值波動度","以ROA估計之mu"];
numeric=[V Dpre VEpre sigmaA sigmaEpre mupre];
company1=["公司名稱";"2880 華南金";"1605 華新";"2610 華航";"6505 台塑化";"2883 開發金";"2603 長榮";"1717 長興";"2892 第一金";"3045 台灣大";"9904 寶成";"9907 統一實";"5820 日盛金";"2807 渣打銀行";"2808 北商銀";"2834 臺企銀";"2884 玉山金";"2885 元大金(復華)";"2891 中信金";"2890 永豐金"];
company2=[TitleM;numeric];
company=[company1 company2];
%寫入excel
xlswrite("D:/python/Vasicek matlab.xlsx",company,"公司價值")
xlswrite("D:/python/Vasicek matlab.xlsx",ARmatrix,"AR(1)")
xlswrite("D:/python/Vasicek matlab.xlsx",VSmatrix,"Vasicek參數估計")
xlswrite("D:/python/Vasicek matlab.xlsx",Spreadmatrix,"Spread")
xlswrite("D:/python/Vasicek matlab.xlsx",LcovM,"LIBOR共變異數矩陣")
xlswrite("D:/python/Vasicek matlab.xlsx",CPcovM,"CP共變異數矩陣")
xlswrite("D:/python/Vasicek matlab.xlsx",CP180covM,"CP180共變異數矩陣")
xlswrite("D:/python/Vasicek matlab.xlsx",CMScovM,"CMS共變異數矩陣")
xlswrite("D:/python/Vasicek matlab.xlsx",RcovM,"無風險利率共變異數矩陣")