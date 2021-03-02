import numpy as np
import math
import scipy.stats as stats
import datetime
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
starttime=datetime.datetime.now()
def interdata(io,sheet,column):
    #載入data,輸出樣本矩陣,io是路徑,sheet是工作表編號,column是樣本欄位
    sheet=pd.read_excel(io,sheet_name=sheet,usecols=column)
    sheet=sheet.values#將dataframe轉成array
    return sheet
def regress(io):
    #AR(1)自回歸,解出alpha,beta,RMSE,資料筆數k,io是路徑
    rt=pd.read_excel(io,index_col=0)#讀取
    rt['lag_value'] = rt.value.shift(1)#創建lag項
    Y=rt.value#呼叫value column
    Y=Y.values#由dataframe轉成array
    X=rt.lag_value
    X=X.values
    X1=sm.add_constant(X)#增加常數項
    model=sm.OLS(Y,X1,missing="drop")
    results=model.fit()
    k=np.size(Y)
    alpha,beta=results.params
    wherenan=np.isnan(X)
    X[wherenan]=0
    ypred=alpha+beta*X
    MSE=mean_squared_error(Y,ypred)
    RMSE=MSE**0.5
    return alpha,beta,RMSE,k
def Vasicek(alpha,beta,RMSE,P,k,rt):
    #利用AR(1)所得之參數作為參數起始值解出Vasicek模型所需參數
    #alpha,beta,RMSE為AR(1)的估計參數,P是切割數,k是data的時間總長度,rt是利率的樣本
    L=1#pdf起始值
    dt=1/12#利率每月變動
    #依據AR(1)的資訊推導參數起始值
    sigmar=RMSE/(dt**0.5)
    a=(1-beta)*dt
    mur=alpha/(a*dt)
    #將a,mu,sigma從參數起始值的0.01倍到10倍做等間格切割矩陣
    #a沿X軸等間隔切割,mu沿Y軸,sigma沿Z軸
    apre1=np.linspace(0.01*a,10*a,P,dtype=np.float64)
    apre2=apre1.T
    apre3=np.repeat(apre2[:,np.newaxis],P,axis=1)
    aM=np.repeat(apre3[:,:,np.newaxis],P,axis=2)
    murpre1=np.linspace(0.01*mur,10*mur,P,dtype=np.float64)
    murpre2=np.tile(murpre1,[P,1])
    murM=np.repeat(murpre2[:,:,np.newaxis],P,axis=2)
    sigmarpre1=np.linspace(0.01*sigmar,10*sigmar,P,dtype=np.float64).reshape(1,1,P)
    sigmarpre2=np.repeat(sigmarpre1,P,axis=1)
    sigmarM=np.repeat(sigmarpre2,P,axis=0)
    #代入最大概似函數
    for i in range(k-1):
        pdf=(1/((math.pi*(2*sigmarM**2)*dt)**0.5))*np.exp(-((((rt[i+1,0]-rt[i,0])-aM*(murM-rt[i,0])*dt))**2)/(2*((sigmarM**2)*dt)))
        if i>0:
            L=L*pdf
        else:
            L=pdf
    #求最大概似函數為最大時的各個參數,即為使用的參數估計值
    maxx=np.argmax(L)
    rowr=int(maxx/(P*P))
    columnr=int((maxx-(P*P*rowr))/P)
    third=(maxx-(P*P*rowr))%P
    cala=aM[rowr,columnr,third]
    calmur=murM[rowr,columnr,third]
    calsigmar=sigmarM[rowr,columnr,third]
    return cala,calmur,calsigmar#輸出Vasicek模型的三個參數
def interest(rstart,a,mur,sigmar,dW,deltatm,T,Q):
    #利用參數估計值模擬Q條利率的路徑
    #rstart是利率起始值,a,mur,sigmar是Vasicek的參數估計值,dW是winner process,deltatm付息日間隔,T模擬時間長度,Q是模擬次數
    t=np.full((1,T,Q),deltatm)#付息日的累加矩陣
    dt=np.cumsum(t,axis=1)#累加時間和winner process
    dWt=np.cumsum(dW,axis=1)
    drt=a*(mur-rstart)*dt+sigmar*dWt#代入Vasicek公式
    rt=rstart+drt#得到每期利率模擬值
    return rt
def forrange(a,b,c,N,Q):
    #將一維等差矩陣依照廠商家數N,模擬次數Q擴展至三維
    #a是一維等差矩陣的起始點,b是終點,c是每次增加的數,N是廠商家數,Q是模擬次數
    forrange=np.arange(a,b,c)#建立一個以a為起點,以b為終點,每次增加c的一維等差矩陣
    forrange=np.tile(forrange,[N,1])#擴展至二維
    forrange=np.repeat(forrange[:,:,np.newaxis],Q,axis=2)#擴展至三維
    #axis=0時矩陣以y軸擴展,=1時矩陣以x軸擴展,=3時以z軸擴展
    return forrange
def common(a,T,Q):
    #將一維矩陣a依照時間數T,模擬次數Q擴展成N*T*Q的三維矩陣
    common=np.repeat(a,T,axis=1)#矩陣擴展至二維N*T
    common=np.repeat(common[:,:,np.newaxis],Q,axis=2)#擴展為三維N*T*Q
    return common
def Loop(a,b,c,Zi):
    #資產池內有些公司發行多個債券,因此計算損失額度時同家公司發行的債券需要使用同組參數
    #a是債券的起始編號,b是債券的截止編號,c是發行公司的編號,Zi是隨機變數的矩陣
    for i  in range(a,b):#將公司資產V,負債D,波動度sigma,mu依照編號分配到各自的位置
        Vt[i,0]=V[c,0]
        Dt[i,0]=Dpre[c]
        sigma[i,0]=sigmaA[c,0]
        mu[i,0]=mupre[c,0]
    Zpre1=Zim[c,:,:]#將隨機變數的矩陣依照編號分配到各自的位置
    Zpre2=np.tile(Zpre1,[b-a,1,1])
    Zi=Zi
    #Zi如果為0則沒有需合併的矩陣
    if a!=0:
        Zi=np.vstack((Zi,Zpre2))
    else:
        Zi=Zpre2
    return Vt,Dt,sigma,mu,Zi#輸出公司資產矩陣Vt,負債矩陣Dt,波動度矩陣sigma,mu矩陣,隨機變數矩陣Zi
def Pool(HU,HL,Ltm):
    #計算CDO的各期分券的分券矩陣剩餘多少本金,HU是分券上界,Hl是分券下界,Ltm是損失額度
    rg=HU-HL#rg是分券上下界的差距
    Pjtm=HU-Ltm#Pjtm是j分券的剩餘本金
    Pjtm[Pjtm<0]=0#剩餘本金若小於0代表分券本金全數損失
    Pjtm[Pjtm>rg]=rg#剩餘本金大於分券發行額代表分券本金無損
    return Pjtm
def Sj0(sumrptm,tm,Pjtm,Ptm,r,Q):
    #CDO的評價公式,計算公平利率
    #sumrptm是支付利率矩陣,tm是付息日矩陣,Pjtm是j分券剩餘本金矩陣,Ptm是總剩餘本金矩陣,r是無風險利率矩陣,Q是模擬次數
    UpSj0=sumrptm*np.exp(-r*tm)*Pjtm/Ptm#將評價的公式分成分子分母最後再相除,Upsj0為分子,DownSj0為分母
    Pjtm[Pjtm<=0]=1#防止除以0
    DownSj0=Pjtm*np.exp(-r*tm)
    sumUpSj0=np.sum(UpSj0,axis=0,dtype=np.float64)#沿著Y軸加總後矩陣剩下1*Q維
    sumDownSj0=np.sum(DownSj0,axis=0,dtype=np.float64)
    Sj0=sumUpSj0/sumDownSj0
    Sj0=np.sum(Sj0,dtype=np.float64)#加總後除以模擬次數即得公平利率
    Sj0=Sj0/Q
    return Sj0
def Sj1(deltatm,r,tm,Pjtm,sumDl,Q):
    #CDO的評價公式,計算來自資產池的利息
    #deltatm是付息日間隔,r是無風險利率矩陣,tm是付息日矩陣,Pjtm是j分券剩餘本金矩陣,sumDl是累積損失矩陣,Q是模擬次數
    SjPL=deltatm*np.exp(-r*tm)*Pjtm
    sumSjPL=np.sum(SjPL,axis=0)
    Sj1=sumDl/sumSjPL
    Sj1=np.sum(Sj1)#加總後除以模擬次數即得應付利率
    return Sj1/Q
def translist(Aarray,M):
    #將要輸出的資訊轉置矩陣後儲存成list,Aarray是欲輸出的矩陣資訊,M是廠商家數
    Aarray1=Aarray.reshape(M)
    Alist=list(Aarray1)
    return Alist
#Vasicek參數設定
P=100#切割數
#Ronn and Verma參數設定
M=19#廠商家數
TimeS=1#時間長度
S=5000#切割數
#Pricing參數設定
Q=10000#模擬次數
N=37#債券個數
T=27#時間切割段數
Time=6.75#模擬時間長數
tm1=forrange(Time/T,Time+Time/T,Time/T,N,Q)#付息日矩陣
deltatm=Time/T#每段時間間隔
#Vasicek data,匯入資料後進行AR(1)回歸作為參數起始值後,估計Vasicek模型的參數
LIBORrt=interdata("D:/paper/vasicek/6ML 5y.xls",0,[2])#匯入利率資料(路徑,工作表編號,樣本欄位)
Lalpha,Lbeta,LRMSE,Lk=regress("D:/paper/vasicek/6ML 5y.xls")#AR(1)回歸後得到alpha,beta,mse
LIBORa,LIBORmur,LIBORsigmar=Vasicek(Lalpha,Lbeta,LRMSE,P,Lk,LIBORrt)#以AR(1)的參數作為參數估計起始值估計Vasicek的a,mu,sigma
CMSrt=interdata("D:/paper/vasicek/5CMS-2CMS 5y.xlsx",0,[2])
CMSalpha,CMSbeta,CMSRMSE,CMSk=regress("D:/paper/vasicek/5CMS-2CMS 5y.xlsx")
CMSa,CMSmur,CMSsigmar=Vasicek(CMSalpha,CMSbeta,CMSRMSE,P,CMSk,CMSrt)
CPrt=interdata("D:/paper/vasicek/90CP 5y.xlsx",0,[2])
CPalpha,CPbeta,CPRMSE,CPk=regress("D:/paper/vasicek/90CP 5y.xlsx")
CPa,CPmur,CPsigmar=Vasicek(CPalpha,CPbeta,CPRMSE,P,CPk,CPrt)
CP180rt=interdata("D:/paper/vasicek/180CP 5y.xlsx",0,[2])
CP180alpha,CP180beta,CP180RMSE,CP180k=regress("D:/paper/vasicek/180CP 5y.xlsx")
CP180a,CP180mur,CP180sigmar=Vasicek(CP180alpha,CP180beta,CP180RMSE,P,CP180k,CP180rt)
rrt=interdata("D:/paper/vasicek/郵局1個月定存利率 5y.xlsx",0,[2])
ralpha,rbeta,rRMSE,rk=regress("D:/paper/vasicek/郵局1個月定存利率 5y.xlsx")
ra,rmur,rsigmar=Vasicek(ralpha,rbeta,rRMSE,P,rk,rrt)
#將利率及模擬公司資產的隨機變數先進行模擬
Z=np.random.normal(0,deltatm,(M+6,T,Q))#Z~N(0,deltatm)
#將隨機變數的矩陣分成模擬公司資產用的與模擬利率用的
Zim=Z[0:M,:,:]#公司特有因子
Zm=Z[M:M+1,:,:]#市場系統因子
Zm=np.tile(Zm,[N,1,1])
#將利率的隨機變數從矩陣取出並模擬利率的變化
LIBORdW=Z[M+1:M+2,:,:]#從矩陣取出
LIBOR=interest(LIBORrt[Lk-1,0],LIBORa,LIBORmur,LIBORsigmar,LIBORdW,deltatm,T,Q)#模擬利率變化
CPdW=Z[M+2:M+3,:,:]
CP=interest(CPrt[CPk-1,0],CPa,CPmur,CPsigmar,CPdW,deltatm,T,Q)
CP180dW=Z[M+3:M+4,:,:]
CP180=interest(CP180rt[CP180k-1,0],CP180a,CP180mur,CP180sigmar,CP180dW,deltatm,T,Q)
CMSdW=Z[M+4:M+5,:,:]
CMS=interest(CMSrt[CMSk-1,0],CMSa,CMSmur,CMSsigmar,CMSdW,deltatm,T,Q)
rdW=Z[M+5:M+6,:,:]
rpre=interest(rrt[rk-1,0],ra,rmur,rsigmar,rdW,deltatm,T,Q).reshape(T,Q)
#模擬出的負利率使其等於0
LIBOR[LIBOR<0]=0
CP[CP<0]=0
CMS[CMS<0]=0
CP[CP180<0]=0
rpre[rpre<0]=0
r=np.cumsum(rpre,axis=0)#將無風險利率累加作為每一期的折現率
#匯入公司資訊與資產池內債券資訊
data1=interdata("D:/python/新2005年玉山資產池概要.xlsm","程式用",[0,1,2,3,4])
#依序為公司名稱,負債,權益波動度,權益價值,mu
company,Dpre,sigmaEpre,VEpre,mupre=np.hsplit(data1,[1,2,3,4])
data2=interdata("D:/python/新2005年玉山資產池概要.xlsm","程式用2",[0,1])
#Bi是各債券本金,irate是各債券之固定利率部分
Bi,irate=np.hsplit(data2,2)
#預先建立一個公司資產價值與資產波動度的M*1的0矩陣以便計算後填入
V=np.zeros([M,1])
sigmaA=np.zeros([M,1])
rx=rrt[rk-1,0]#Ronn and Verma的折現率
#將權益價值VE及權益波動度sigmaE逐筆代入
for i in range(M):
    VE=VEpre[i,0]
    sigmaE=sigmaEpre[i,0]
    D=Dpre[i,0]
    #VA為1倍(VE+D)至10倍(VE+D)依X軸排序
    VApre1=np.linspace(1*(VE+D),10*(VE+D),S)
    VApre2=np.tile(VApre1,[S,1])
    #sigmaA為0.01倍sigmaE至10倍sigmaE依Y軸排序
    sigmaApre1=np.linspace(0.01*sigmaE,10*sigmaE,S).reshape(1,S)
    sigmaApre2=sigmaApre1.T
    sigmaAarray=np.repeat(sigmaApre2,S,axis=1)
    #代入公式
    d1pre=np.log(VApre2/D)
    d1=(d1pre+(rx+(sigmaAarray**2)/2)*TimeS)/(sigmaAarray*(TimeS**0.5))
    d2=d1-sigmaAarray*(TimeS**0.5)
    Nd1=stats.norm.cdf(d1)
    Nd2=stats.norm.cdf(d2)
    #解Ronn and Verma的聯立方程式
    apre=VApre2*Nd1-D*np.exp(-rx*TimeS)*Nd2-VE
    bpre=VApre2/VE*Nd1*sigmaAarray-sigmaE
    a=apre/VE#利用誤差百分比的概念
    b=bpre/sigmaE
    c=abs(a)+abs(b)
    e=np.argmin(c)#找出最小誤差百分比的參數座標位置
    row=int(e/S)#將位置除以矩陣column去掉小數點後計算在第幾個row
    column=e%S#餘數即為column
    V[i,0]=VApre2[row,column]
    sigmaA[i,0]=sigmaAarray[row,column]
#Pricing data,先建立各參數的0矩陣
rho=np.zeros([N,1])
Dt=np.zeros([N,1])
Vt=np.zeros([N,1])
sigma=np.zeros([N,1])
mu=np.zeros([N,1])
#將公司的參數分配至期發行的債券
Zi=0
Vt,Dt,sigma,mu,Zi=Loop(0,3,0,Zi)
Vt,Dt,sigma,mu,Zi=Loop(3,5,1,Zi)
Vt,Dt,sigma,mu,Zi=Loop(5,9,2,Zi)
Vt,Dt,sigma,mu,Zi=Loop(9,12,3,Zi)
Vt,Dt,sigma,mu,Zi=Loop(12,14,4,Zi)
Vt,Dt,sigma,mu,Zi=Loop(14,15,5,Zi)
Vt,Dt,sigma,mu,Zi=Loop(15,17,6,Zi)
Vt,Dt,sigma,mu,Zi=Loop(17,19,7,Zi)
Vt,Dt,sigma,mu,Zi=Loop(19,20,8,Zi)
Vt,Dt,sigma,mu,Zi=Loop(20,22,9,Zi)
Vt,Dt,sigma,mu,Zi=Loop(22,24,10,Zi)
Vt,Dt,sigma,mu,Zi=Loop(24,26,11,Zi)
Vt,Dt,sigma,mu,Zi=Loop(26,27,12,Zi)
Vt,Dt,sigma,mu,Zi=Loop(27,30,13,Zi)
Vt,Dt,sigma,mu,Zi=Loop(30,32,14,Zi)
Vt,Dt,sigma,mu,Zi=Loop(32,33,15,Zi)
Vt,Dt,sigma,mu,Zi=Loop(33,34,16,Zi)
Vt,Dt,sigma,mu,Zi=Loop(34,35,17,Zi)
Vt,Dt,sigma,mu,Zi=Loop(35,36,1,Zi)
Vt,Dt,sigma,mu,Zi=Loop(36,37,18,Zi)
#將參數的矩陣擴展成三維
Bi=common(Bi,T,Q)
rho=common(rho,T,Q)
mu=common(mu,T,Q)
sigma=common(sigma,T,Q)
Dt=common(Dt,T,Q)
Vt=common(Vt,T,Q)
#債券的利率為固定利率+上一個浮動利率,此處先將固定利率放入矩陣
rConstant,rLIBOR,rCP,rCMS,rMax=np.vsplit(irate,[1,34,35,36])
rConstant=common(rConstant,T,Q)
rLIBOR=common(rLIBOR,T,Q)
rCP=common(rCP,T,Q)
rCMS=common(rCMS,T,Q)
rMax=common(rMax,T,Q)
#利用固定利率矩陣與模擬的利率矩陣計算各債券應支付的利率
LIBORt=rLIBOR-LIBOR
CPt=rCP+CP
CMSt=rCMS+CMS
Max1=LIBOR-CP180#債券為MAX{LIBOR,180CP},因此需判斷LIBOR與180天CP哪個利率較高
Max1[Max1>0]=1
Max1[Max1<0]=0
Max2=1-Max1
Max3=LIBOR*Max1+CP180*Max2
Maxt=rMax+Max3
#將利率矩陣拼接成N*T*Q
two=np.vstack((rConstant,LIBORt))
three=np.vstack((two,CPt))
four=np.vstack((three,CMSt))
five=np.vstack((four,Maxt))
five[five<0]=0
#Pricing計算
dz=(1-rho**2)**0.5*Zi+rho*Zm
expopre=np.exp((mu-1/2*(sigma**2))*deltatm+sigma*dz)#建立exp的矩陣
expo=np.cumprod(expopre,axis=1)#建立累乘exp矩陣
Vtex=Vt*expo#得到各期模擬的公司資產
decide=Vtex-Dt#公司價值減負債的值判斷是否違約
decide[decide>0]=1#若大於0設為1
decide[decide<0]=0#小於0設為0
decide=np.cumsum(decide,axis=1)#以X軸累加
deduction=forrange(0,T,1,N,Q)#建立一個0到T每次增加1的矩陣，兩式相減若小於0即可知道公司何時違約何時存活
survival=decide-deduction
survival[survival<0]=0#讓生存矩陣小於0的部分等於0
rptm=Bi*survival*five#生存矩陣乘以本金和利率可得每期利息
sumrptm=np.sum(rptm,axis=0)#加總利息
defaultpre=(1-survival)#違約矩陣
Ri=Vtex/Dt#違約回復率=違約當期的資產/負債
default=np.cumsum(defaultpre,axis=1)#判斷在哪一期有損失
default[default>1]=0#有損失的那期為1其他為0
Ltm=Bi*(1-Ri)*default#得到損失金額
sumLtmpre=np.sum(Ltm,axis=0,dtype=np.float64)#將所有債券的損失加總,變成T*Q的矩陣
sumLtm=np.cumsum(sumLtmpre,axis=0)#累加損失矩陣,得到各期總損失金額
Dlpre=default*Bi*(1-Ri)*np.exp(-r*tm1)#損失現值
Dl=np.sum(Dlpre,axis=0,dtype=np.float64)#將所有債券的損失現值加總,變成T*Q的矩陣
sumDl=np.sum(Dl,axis=0)#將T期的損失現值加總,用以計算Sj1
tm2=tm1[0:1,:,:].reshape(T,Q)#從付息日矩陣取出1*T*Q的矩陣,將他轉為T*Q
tranche=interdata("D:/python/新2005年玉山資產池概要.xlsm","分券金額",[1])#匯入分券金額data
#計算分券剩餘本金矩陣
P1tm=Pool(tranche[3,0]+tranche[2,0]+tranche[1,0]+tranche[0,0],tranche[3,0]+tranche[2,0]+tranche[1,0],sumLtm)
P2tm=Pool(tranche[3,0]+tranche[2,0]+tranche[1,0],tranche[3,0]+tranche[2,0],sumLtm)
P3tm=Pool(tranche[3,0]+tranche[2,0],tranche[3,0],sumLtm)
P4tm=Pool(tranche[3,0],0,sumLtm)
Ptm=P1tm+P2tm+P3tm+P4tm#加總分券本金
#計算CDO應支付的利率
S10=Sj0(sumrptm,tm2,P1tm,Ptm,r,Q)
S20=Sj0(sumrptm,tm2,P2tm,Ptm,r,Q)
S30=Sj0(sumrptm,tm2,P3tm,Ptm,r,Q)
S40=Sj0(sumrptm,tm2,P4tm,Ptm,r,Q)
print("S10=",S10,"S20=",S20,"S30=",S30,"S40=",S40)
S11=Sj1(deltatm,r,tm2,P1tm,sumDl,Q)
S21=Sj1(deltatm,r,tm2,P2tm,sumDl,Q)
S31=Sj1(deltatm,r,tm2,P3tm,sumDl,Q)
S41=Sj1(deltatm,r,tm2,P4tm,sumDl,Q)
print("S11=",S11,"S21=",S21,"S31=",S31,"S41=",S41)
TS1=S10+S11
TS2=S20+S21
TS3=S30+S31
TS4=S40+S41
print("TS1=",TS1,"TS2=",TS2,"TS3=",TS3,"TS4=",TS4)
#將要輸出的資料轉為list
Vlist=translist(V,M)
Dlist=translist(Dpre,M)
VElist=translist(VEpre,M)
mulist=translist(mupre,M)
sigmaElist=translist(sigmaEpre,M)
sigmaAlist=translist(sigmaA,M)
companylist=translist(company,M)
#輸出資料至EXCEL
titlelist=["LIBOR","5CMS-2CMS","90CP","180CP","無風險利率"]#把要輸出的list數據對準標題列
ARlist1=[Lalpha,CMSalpha,CPalpha,CP180alpha,ralpha]
ARlist2=[Lbeta,CMSbeta,CPbeta,CP180beta,rbeta]
ARlist3=[LRMSE,CMSRMSE,CPRMSE,CP180RMSE,rRMSE]
Vasiceklist1=[LIBORa,CMSa,CPa,CP180a,ra]
Vasiceklist2=[LIBORmur,CMSmur,CPmur,CP180mur,rmur]
Vasiceklist3=[LIBORsigmar,CMSsigmar,CPsigmar,CP180sigmar,rsigmar]
value1=["S10",S10,"S20",S20,"S30",S30,"S40",S40]
value2=["S11",S11,"S21",S21,"S31",S31,"S41",S41]
value3=["TS1",TS1,"TS2",TS2,"TS3",TS3,"TS4",TS4]
dit1={"公司名稱":companylist,'公司估計價值':Vlist,"公司負債價值":Dlist,"公司權益價值":VElist,"公司總波動度":sigmaAlist,"公司權益價值波動度":sigmaElist,"以ROA估計之mu":mulist}
dit2={'AR':titlelist,"alpha":ARlist1,"Beta":ARlist2,"RMSE":ARlist3}
dit3={"Vasicek":titlelist,"a":Vasiceklist1,"mu":Vasiceklist2,"sigma":Vasiceklist3}
dit4={"from pool":value1,"from loss":value2,"total":value3}
file_path = "D:/python/Vasicek.xlsx"
writer = pd.ExcelWriter(file_path)#開始寫入
#把dict轉成dataframe輸出至excel
df1=pd.DataFrame(dit1)
df2=pd.DataFrame(dit2)
df3=pd.DataFrame(dit3)
df4=pd.DataFrame(dit4)
#將表格儲存至各自的工作表
df1.to_excel(writer, columns=["公司名稱",'公司估計價值',"公司負債價值","公司權益價值","公司總波動度","公司權益價值波動度","以ROA估計之mu"], index=False,sheet_name='公司價值')
df2.to_excel(writer, columns=['AR',"alpha","Beta","RMSE"], index=False,sheet_name='AR(1)')
df3.to_excel(writer, columns=["Vasicek","a","mu","sigma"], index=False,sheet_name='Vasicek estimate')
df4.to_excel(writer, columns=["from pool","from loss","total"], index=False,sheet_name='Spread')
writer.save()#儲存
endtime=datetime.datetime.now()
print (endtime-starttime)
