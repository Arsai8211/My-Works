function[Pjtm]=Pool(HU,HL,sumLtm)
%�p��CDO���U�����骺����x�}�Ѿl�h�֥���(����W��,����U��,�l���B��)
range=HU-HL;%rg�O����W�U�ɪ��t�Z
Pjtm=HU-sumLtm;%Pjtm�Oj���骺�Ѿl����
Pjtm(Pjtm<0)=0;%�Ѿl�����Y�p��0�N����饻�����Ʒl��
Pjtm(Pjtm>range)=range;%�Ѿl�����j�����o���B�N����饻���L�l






