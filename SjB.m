function[SjB]=SjB(deltatm,r,tm,Pjtm,sumDl,Q)
%CDO����������,�p��Ӧ۸겣�����Q��,(�I���鶡�j,�L���I�Q�v�x�},�I����x�},j����Ѿl�����x�},�ֿn�l���x�},��������)
SjPL=deltatm.*exp(-r.*tm).*Pjtm;
sumSjPL=sum(SjPL,2);
SjB=sumDl./sumSjPL;
SjB=sum(SjB,3);%�[�`�ᰣ�H�������ƧY�o���I�Q�v
SjB=SjB./Q;