function[alpha,beta,mse,k]=ARone(filename)
%AR(1)�ۦ^�k,�ѥXalpha,beta,RMSE,��J�Q�vdata�i��AR(1)�ѼƦ��p
rt=inputdata(filename,1,"value");%��J���
[k,q]=size(rt);%�d�߸�Ʀ��h�ֵ����
lagone=zeros(k,q);%�إ߸������0�x�}
%�إ߸�����x�}���j��
for i=1:k
    if i==1
        lagone(i,1)=nan;
    else
        lagone(i,1)=rt(i-1,1);
    end
end
%�H�̤p����k�^�k
stats=regstats(rt,lagone);
coef=getfield(stats,"beta");
%��X�Ѽ�alpha,beta,mse
mse=getfield(stats,"mse");
alpha=coef(1,1);
beta=coef(2,1);