%Vasicek�ѼƳ]�w
P=100;%#Vasicek�x�}���μ�
%Ronn and Verma�ѼƳ]�w
M=19;%�t�Ӯa��
TimeS=1;%���pRonn and Verma�ϥΪ��ɶ��Ѽ�
S=5000;%���μ�
%Pricing�ѼƳ]�w
Q=10000;%��������
N=37;%�Ũ�Ӽ�
T=27;%�ɶ����άq��
Time=6.75;%�����ɶ�����
tm1=custom3D(Time/T,Time,T,N,Q);%�I����x�}
deltatm=Time/T;%�C�q�ɶ����j
%Vasicek data,�פJ��Ʈƫ�i��AR(1)�^�k�@���Ѽư_�l�ȫ�,���pVasicek�ҫ����Ѽ�
LIBORrt=inputdata("D:/paper/vasicek/6ML 5y.xls",1,"value");%�פJ�Q�v���(���|,�u�@��,���D�W��)
[Lalpha,Lbeta,Lmse,Lk]=ARone("D:/paper/vasicek/6ML 5y.xls");%AR(1)�^�k��o��alpha,beta,mse
[La,Lmur,Lsigmar]=Vasicek(Lalpha,Lbeta,Lmse,P,Lk,LIBORrt);%�HAR(1)���ѼƧ@���ѼƦ��p�_�l�Ȧ��pVasicek��a,mu,sigma
CMSrt=inputdata("D:/paper/vasicek/5CMS-2CMS 5y.xlsx",1,"value");
[CMSalpha,CMSbeta,CMSmse,CMSk]=ARone("D:/paper/vasicek/5CMS-2CMS 5y.xlsx");
[CMSa,CMSmur,CMSsigmar]=Vasicek(CMSalpha,CMSbeta,CMSmse,P,CMSk,CMSrt);
CPrt=inputdata("D:/paper/vasicek/90CP 5y.xlsx",1,"value");
[CPalpha,CPbeta,CPmse,CPk]=ARone("D:/paper/vasicek/90CP 5y.xlsx");
[CPa,CPmur,CPsigmar]=Vasicek(CPalpha,CPbeta,CPmse,P,CPk,CPrt);
CP180rt=inputdata("D:/paper/vasicek/180CP 5y.xlsx",1,"value");
[CP180alpha,CP180beta,CP180mse,CP180k]=ARone("D:/paper/vasicek/180CP 5y.xlsx");
[CP180a,CP180mur,CP180sigmar]=Vasicek(CP180alpha,CP180beta,CP180mse,P,CP180k,CP180rt);
rrt=inputdata("D:/paper/vasicek/�l��1�Ӥ�w�s�Q�v 5y.xlsx",1,"value");
[ralpha,rbeta,rmse,rk]=ARone("D:/paper/vasicek/�l��1�Ӥ�w�s�Q�v 5y.xlsx");
[ra,rmur,rsigmar]=Vasicek(ralpha,rbeta,rmse,P,rk,rrt);
%�N�Q�v�μ������q�겣���H���ܼƥ��i�����
Z=normrnd(0,deltatm,M+6,T,Q);%Z~N(0,deltatm)
%�N�H���ܼƪ��x�}�����������q�겣�Ϊ��P�����Q�v�Ϊ�
Zim=Z(1:M,:,:);%���q�S���]�l
Zm=Z(M+1,:,:);%�����t�Φ]�l
Zm=repmat(Zm,N,1,1);%�N�����t�Φ]�l���x�}�X�i��N*T*Q
%�N�Q�v���H���ܼƱq�x�}���X�ü����Q�v���ܤ�
LIBORdW=Z(M+2,:,:);%�q�x�}���X
LIBORM=interest(LIBORrt(Lk,1),La,Lmur,Lsigmar,LIBORdW,deltatm,Time,T,Q);%�����Q�v�ܤ�
CPdW=Z(M+3,:,:);
CPM=interest(CPrt(CPk,1),CPa,CPmur,CPsigmar,CPdW,deltatm,Time,T,Q);
CP180dW=Z(M+4,:,:);
CP180M=interest(CP180rt(CP180k,1),CP180a,CP180mur,CP180sigmar,CP180dW,deltatm,Time,T,Q);
CMSdW=Z(M+5,:,:);
CMSM=interest(CMSrt(CMSk,1),CMSa,CMSmur,CMSsigmar,CMSdW,deltatm,Time,T,Q);
rdW=Z(M+6,:,:);
rpre=interest(rrt(rk,1),ra,rmur,rsigmar,rdW,deltatm,Time,T,Q);
%�����X���t�Q�v�Ϩ䵥��0
LIBORM(LIBORM<0)=0;
CPM(CPM<0)=0;
CP180M(CP180M<0)=0;
CMSM(CMSM<0)=0;
rpre(rpre<0)=0;
r=cumsum(rpre,1);%�N�L���I�Q�v�֥[�@���C�@������{�v
%�פJ���q��T�P�겣�����Ũ��T
%�̧Ǭ��t��,�v�q�i�ʫ�,�v�q����,mu,����,�U�Ũ餧�T�w�Q�v
Dpre=inputdata("D:/python/�s2005�~�ɤs�겣�����n.xlsm","�{����","�t��");
sigmaEpre=inputdata("D:/python/�s2005�~�ɤs�겣�����n.xlsm","�{����","�v�q�i�ʫ�");
VEpre=inputdata("D:/python/�s2005�~�ɤs�겣�����n.xlsm","�{����","�v�q����");
mupre=inputdata("D:/python/�s2005�~�ɤs�겣�����n.xlsm","�{����","ROA�|�ᮧ�e");
Bi=inputdata("D:/python/�s2005�~�ɤs�겣�����n.xlsm","�{����2","����");
irate=inputdata("D:/python/�s2005�~�ɤs�겣�����n.xlsm","�{����2","��¦�Q�v");
rx=rrt(rk,1);%Ronn and Verma����{�v
%�w���إߤ@�Ӥ��q�겣���ȻP�겣�i�ʫת�M*1��0�x�}�H�K�p����J
V=zeros(M,1);
sigmaA=zeros(M,1);
%�N�v�q����VE���v�q�i�ʫ�sigmaE�v���N�J
for i=1:M
    VE=VEpre(i,1);
    sigmaE=sigmaEpre(i,1);
    D=Dpre(i,1);
    %VA��1��(VE+D)��10��(VE+D)��X�b�Ƨ�
    VApre1=linspace(1*(VE+D),10*(VE+D),S);
    VApre2=repmat(VApre1,S,1);
    %sigmaA��0.01��sigmaE��10��sigmaE��Y�b�Ƨ�
    sigmaApre1=linspace(0.01*sigmaE,10*sigmaE,S);
    sigmaAarray=repmat(sigmaApre1',1,S);
    %�N�J����
    d1pre=log(VApre2./D);
    d1=(d1pre+(rx+(sigmaAarray^2)/2).*TimeS)./(sigmaAarray.*(TimeS^0.5));
    d2=d1-sigmaAarray*(TimeS^0.5);
    Nd1=normcdf(d1);
    Nd2=normcdf(d2);
    %��Ronn and Verma���p�ߤ�{��
    apre=VApre2.*Nd1-D.*exp(-rx.*TimeS).*Nd2-VE;
    bpre=VApre2./VE.*Nd1.*sigmaAarray-sigmaE;
    a=apre./VE;%�Q�λ~�t�ʤ��񪺷���
    b=bpre./sigmaE;
    c=abs(a)+abs(b);
    where=min(min(c));%��X�̤p�~�t�ʤ��񪺰ѼƮy�Ц�m
    [row,column]=find(c==where);
    V(i,1)=VApre2(row,column);
    sigmaA(i,1)=sigmaAarray(row,column);
end
%Pricing data,���إߦU�Ѽƪ�0�x�}
Vt=zeros(N,1);
Dt=zeros(N,1);
sigma=zeros(N,1);
mu=zeros(N,1);
rho=zeros(N,1);
%�N���q���ѼƤ��t�ܴ��o�檺�Ũ�
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
%�N�Ѽƪ��x�}�X�i���T��
Bi=threeD(Bi,T,Q);
rho=threeD(rho,T,Q);
mu=threeD(mu,T,Q);
sigma=threeD(sigma,T,Q);
Dt=threeD(Dt,T,Q);
Vt=threeD(Vt,T,Q);
%�Ũ骺�Q�v���T�w�Q�v+�W�@�ӯB�ʧQ�v,���B���N�T�w�Q�v��J�x�}
rConstant=irate(1,:,:);
rLIBOR=irate(2:34,:,:);
rCP=irate(35,:,:);
rCMS=irate(36,:,:);
rMax=irate(37,:,:);
%�T�w�Q�v�x�}�X�i���T��
rConstant=threeD(rConstant,T,Q);
rLIBOR=threeD(rLIBOR,T,Q);
rCP=threeD(rCP,T,Q);
rCMS=threeD(rCMS,T,Q);
rMax=threeD(rMax,T,Q);
%�Q�ΩT�w�Q�v�x�}�P�������Q�v�x�}�p��U�Ũ�����I���Q�v
LIBORt=rLIBOR-LIBORM;
CPt=rCP+CPM;
CMSt=rCMS+CMSM;
Max1=LIBORM-CP180M;%�Ũ鬰MAX{LIBOR,180CP},�]���ݧP�_LIBOR�P180CP���ӧQ�v����
Max1(Max1>0)=1;
Max1(Max1<0)=0;
Max2=1-Max1;
Max3=LIBORM.*Max1+CP180M.*Max2;
Maxt=rMax+Max3;
%�N�Q�v�x�}������N*T*Q
two=[rConstant;LIBORt];
three=[two;CPt];
four=[three;CMSt];
five=[four;Maxt];
five(five<0)=0;
%Pricing�p��
dz=((1-(rho.^2)).^0.5).*Zi+rho.*Zm;
expopre=exp((mu-1/2.*(sigma.^2)).*deltatm+sigma.*dz);%�إ�exp���x�}
expo=cumprod(expopre,2);%�إ߲֭�exp�x�}(dim=1�Hy�b�֭�,=2�Hx�b�֭�)
Vtex=Vt.*expo;%�o��U�����������q�겣
decide=Vtex-Dt;%���q���ȴ�t�Ū��ȧP�_�O�_�H��
decide(decide>0)=1;%�Y�j��0�]��1
decide(decide<0)=0;%�p��0�]��0
decide=cumsum(decide,2);%�HX�b�֥[
deduction=custom3D(0,T-1,T,N,Q);%�إߤ@��0��T�C���W�[1���x�}�A�⦡�۴�Y�p��0�Y�i���D���q��ɹH����ɦs��
survival=decide-deduction;
survival(survival<0)=0;%���ͦs�x�}�p��0����������0
rptm=Bi.*survival.*five;%�ͦs�x�}���H�����M�Q�v�i�o�C���Q��
sumrptm=sum(rptm,1);%�[�`�Q��
defaultpre=(1-survival);%�H���x�}
Ri=Vtex./Dt;%�H���^�_�v=�H��������겣/�t��
default=cumsum(defaultpre,2);%�P�_�b���@�����l��
default(default>1)=0;%���l����������1��L��0
Ltm=Bi.*(1-Ri).*default;%�o��l�����B
sumLtmpre=sum(Ltm,1);%�N�Ҧ��Ũ骺�l���[�`,�ܦ�T*Q���x�}
sumLtm=cumsum(sumLtmpre,1);%�֥[�l���x�},�o��U���`�l�����B
Dlpre=default.*Bi.*(1-Ri).*exp(-r.*tm1);%�l���{��
Dl=sum(Dlpre,1);%�N�Ҧ��Ũ骺�l���{�ȥ[�`,�ܦ�T*Q���x�}
sumDl=sum(Dl,2);%�NT�����l���{�ȥ[�`,�ΥH�p��Sj1
tm2=tm1(1,:,:);%�q�I����x�}���XT*Q���x�}
principal=inputdata("D:/python/�s2005�~�ɤs�겣�����n.xlsm","������B","�o����B");%�פJ������Bdata
%pA��pD�OA.B.C.D�U���骺����W�����B
pA=principal(4,1)+principal(3,1)+principal(2,1)+principal(1,1);
pB=principal(4,1)+principal(3,1)+principal(2,1);
pC=principal(4,1)+principal(3,1);
pD=principal(4,1);
%�p��C������Ѿl����
P1tm=Pool(pA,pB,sumLtm);
P2tm=Pool(pB,pC,sumLtm);
P3tm=Pool(pC,pD,sumLtm);
P4tm=Pool(pD,0,sumLtm);
Ptm=P1tm+P2tm+P3tm+P4tm;%�[�`���饻��
%�p��CDO����I���Q�v
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
%�p��̤j�����k�����ĩ�
RFparameter=[ra,rmur,rsigmar];%���N���p�X�Ӫ��ѼƱƧ�
Lparameter=[La,Lmur,Lsigmar];
CPparameter=[CPa,CPmur,CPsigmar];
CP180parameter=[CP180a,CP180mur,CP180sigmar];
CMSparameter=[CMSa,CMSmur,CMSsigmar];
Lcov=mlecovARone(Lparameter,LIBORrt,@Vasicekpdf);%��J�Ѽ�,�Q�v�˥�,pdf�o��@�ܲ��Ưx�}
CPcov=mlecovARone(CPparameter,CPrt,@Vasicekpdf);
CP180cov=mlecovARone(CP180parameter,CP180rt,@Vasicekpdf);
CMScov=mlecovARone(CMSparameter,CMSrt,@Vasicekpdf);
Rcov=mlecovARone(RFparameter,rrt,@Vasicekpdf);
%��n��X����ƩM���D�ƧǦn
LcovM=["","LIBOR variance-covariance Matrix","";Lcov];
CPcovM=["","CP variance-covariance Matrix","";CPcov];
CP180covM=["","CP180 variance-covariance Matrix","";CP180cov];
CMScovM=["","CMS variance-covariance Matrix","";CMScov];
RcovM=["","Rf variance-covariance Matrix","";Rcov];
ARmatrix=["AR","alpha","Beta","MSE";"LIBOR",Lalpha,Lbeta,Lmse;"5CMS-2CMS",CMSalpha,CMSbeta,CMSmse;"90CP",CPalpha,CPbeta,CPmse;"180CP",CP180alpha,CP180beta,CP180mse;"�L���I�Q�v",ralpha,rbeta,rmse];
VSmatrix=["Vasicek","a","mu","sigma";"LIBOR",La,Lmur,Lsigmar;"5CMS-2CMS",CMSa,CMSmur,CMSsigmar;"90CP",CPa,CPmur,CPsigmar;"180CP",CP180a,CP180mur,CP180sigmar;"�L���I�Q�v",ra,rmur,rsigmar];
Spreadmatrix=["�Ӧ۸겣�����Q�v","�H�l���p�⪺�����Q�v","�`�Q�v";"S10","S11","TS1";S1A,S1B,TS1;"S20","S21","TS2";S2A,S2B,TS2;"S30","S31","TS3";S3A,S3B,TS3;"S40","S41","TS4";S4A,S4B,TS4];
TitleM=['���q���p����',"���q�t�Ż���","���q�v�q����","���q�`�i�ʫ�","���q�v�q���Ȫi�ʫ�","�HROA���p��mu"];
numeric=[V Dpre VEpre sigmaA sigmaEpre mupre];
company1=["���q�W��";"2880 �ثn��";"1605 �طs";"2610 �د�";"6505 �x���";"2883 �}�o��";"2603 ���a";"1717 ����";"2892 �Ĥ@��";"3045 �x�W�j";"9904 �_��";"9907 �Τ@��";"5820 �鲱��";"2807 ���Ȧ�";"2808 �_�ӻ�";"2834 �O����";"2884 �ɤs��";"2885 ���j��(�_��)";"2891 ���H��";"2890 ���ת�"];
company2=[TitleM;numeric];
company=[company1 company2];
%�g�Jexcel
xlswrite("D:/python/Vasicek matlab.xlsx",company,"���q����")
xlswrite("D:/python/Vasicek matlab.xlsx",ARmatrix,"AR(1)")
xlswrite("D:/python/Vasicek matlab.xlsx",VSmatrix,"Vasicek�ѼƦ��p")
xlswrite("D:/python/Vasicek matlab.xlsx",Spreadmatrix,"Spread")
xlswrite("D:/python/Vasicek matlab.xlsx",LcovM,"LIBOR�@�ܲ��Ưx�}")
xlswrite("D:/python/Vasicek matlab.xlsx",CPcovM,"CP�@�ܲ��Ưx�}")
xlswrite("D:/python/Vasicek matlab.xlsx",CP180covM,"CP180�@�ܲ��Ưx�}")
xlswrite("D:/python/Vasicek matlab.xlsx",CMScovM,"CMS�@�ܲ��Ưx�}")
xlswrite("D:/python/Vasicek matlab.xlsx",RcovM,"�L���I�Q�v�@�ܲ��Ưx�}")