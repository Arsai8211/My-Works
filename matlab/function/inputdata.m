function[rtM]=inputdata(io,sheet,keystr)
%將資料輸入至matlab並建立矩陣(檔案位置,工作表,標題名)
[num,txt]=xlsread(io,sheet);%讀取檔案
column=find(strcmp(txt(1,:),keystr));%尋找標題名的column
[row,col]=size(txt);%查詢txt矩陣的大小
[roww,coll]=size(num);%查詢num矩陣的大小
%若第一個column都不是數據則column需扣掉1
if col-coll==1
    decide=1;
else
    decide=0;
end
rtM=num(:,column-decide);