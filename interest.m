function[rt]=interest(rstart,a,mur,sigmar,dW,deltatm,Time,T,Q)
%�Q�ΰѼƦ��p�ȼ���Q���Q�v�����|(�Q�v�_�l��,a,mu,sigma,winner process,�I���鶡�j,�����ɶ�����,��������)
dt=linspace(deltatm,Time,T);%�I���骺�֥[�x�}
dt=repmat(dt,1,1,Q);%�X�i��1*T*Q�x�}
dWt=cumsum(dW,2);%�֥[winner process
drt=a.*(mur-rstart).*dt+sigmar.*dWt;%�N�JVasicek����
rt=rstart+drt;%�o��C���Q�v������