function[threeD]=threeD(array,T,Q)
%�N�@���x�}array�̷Ӯɶ���T,��������Q�X�i��N*T*Q���T���x�}
threeDpre=repmat(array,1,T);%�x�}�X�i�ܤG��N*T
threeD=repmat(threeDpre,1,1,Q);%�X�i���T��N*T*Q