function[SjA]=SjA(sumrptm,tm,Pjtm,Ptm,r,Q)
%CDO����������,�p�⤽���Q�v,(��I�Q�v�x�},�I����x�},j����Ѿl�����x�},�`�Ѿl�����x�},�L���I�Q�v�x�},��������)
UpSjA=sumrptm.*exp(-r.*tm).*Pjtm./Ptm; %�N�����������������l�����̫�A�۰�,UpSjA�����l,DownSjA������
Pjtm(Pjtm<=0)=1; %����H0
DownSjA=Pjtm.*exp(-r.*tm);
sumUpSjA=sum(UpSjA,2); %�u��Y�b�[�`��x�}�ѤU1*1*Q��
sumDownSjA=sum(DownSjA,2);
SjA=sumUpSjA./sumDownSjA;
SjA=sum(SjA,3);%�[�`�ᰣ�H�������ƧY�o�����Q�v
SjA=SjA./Q;