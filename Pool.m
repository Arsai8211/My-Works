function[Pjtm]=Pool(HU,HL,sumLtm)
%計算CDO的各期分券的分券矩陣剩餘多少本金(分券上界,分券下界,損失額度)
range=HU-HL;%rg是分券上下界的差距
Pjtm=HU-sumLtm;%Pjtm是j分券的剩餘本金
Pjtm(Pjtm<0)=0;%剩餘本金若小於0代表分券本金全數損失
Pjtm(Pjtm>range)=range;%剩餘本金大於分券發行額代表分券本金無損






