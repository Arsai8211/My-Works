function[Vt,Dt,sigma,mu,Zi]=loop(start,ended,company,Zi,Zim,V,D,sigmaA,mupre,Vt,Dt,sigma,mu)
%�겣�������Ǥ��q�o��h�ӶŨ�,�]���p��l���B�׮ɦP�a���q�o�檺�Ũ�ݭn�ϥΦP�հѼ�
%(�Ũ骺�_�l�s��,�Ũ骺�I��s��,�o�椽�q���s��,���t���H���ܼƪ��x�},��l�H���ܼƪ��x�},��l���q�겣�x�},��l���q�t�ůx�},��l�겣�i�ʫׯx�},��lmu�x�},���t�᤽�q�겣�x�},���t�᤽�q�t�ůx�},���t��겣�i�ʫׯx�},���t��mu�x�})
for i=start:ended%�N�H���ܼ�Zim,���q�겣V,�t��D,�i�ʫ�sigmaA,mupre�̷ӽs�����t��U�۪���m
    Vt(i,1)=V(company,1);
    Dt(i,1)=D(company,1);
    sigma(i,1)=sigmaA(company,1);
    mu(i,1)=mupre(company,1);
    Zpre1=Zim(company,:,:);
    Zpre2=repmat(Zpre1,ended-start+1,1,1);
end
%Zi�p�G��0�h�S���ݦX�֪��x�}
if Zi~=0
    Zi=[Zi;Zpre2];
else
    Zi=Zpre2;
end%��X���q�겣�x�}Vt,�t�ůx�}Dt,�i�ʫׯx�}sigma,mu�x�},�H���ܼƯx�}Zi        