function[custom3D]=custom3D(start,ended,divide,N,Q)
%�N�@�����t�x�}�̷Ӽt�Ӯa��N,��������Q�X�i�ܤT��(�_�I,���I,�X����,�t�Ӯa��,��������)
threeDpre1=linspace(start,ended,divide);%�إߤ@�ӥHstart���_�I,�Hended�����I,divide�������@�����t�x�}
threeDpre2=repmat(threeDpre1,N,1);%�X�i�ܤG��
custom3D=repmat(threeDpre2,1,1,Q);%�X�i�ܤT��