function[rtM]=inputdata(io,sheet,keystr)
%�N��ƿ�J��matlab�ëإ߯x�}(�ɮצ�m,�u�@��,���D�W)
[num,txt]=xlsread(io,sheet);%Ū���ɮ�
column=find(strcmp(txt(1,:),keystr));%�M����D�W��column
[row,col]=size(txt);%�d��txt�x�}���j�p
[roww,coll]=size(num);%�d��num�x�}���j�p
%�Y�Ĥ@��column�����O�ƾګhcolumn�ݦ���1
if col-coll==1
    decide=1;
else
    decide=0;
end
rtM=num(:,column-decide);