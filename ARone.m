function[alpha,beta,mse,k]=ARone(filename)
%AR(1)自回歸,解出alpha,beta,RMSE,輸入利率data進行AR(1)參數估計
rt=inputdata(filename,1,"value");%輸入資料
[k,q]=size(rt);%查詢資料有多少筆資料
lagone=zeros(k,q);%建立落後期的0矩陣
%建立落後期矩陣的迴圈
for i=1:k
    if i==1
        lagone(i,1)=nan;
    else
        lagone(i,1)=rt(i-1,1);
    end
end
%以最小平方法回歸
stats=regstats(rt,lagone);
coef=getfield(stats,"beta");
%輸出參數alpha,beta,mse
mse=getfield(stats,"mse");
alpha=coef(1,1);
beta=coef(2,1);