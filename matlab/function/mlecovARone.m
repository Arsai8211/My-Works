function acov = mlecovARone(params,data,pdf)
%輸入paramss矩陣，data矩陣，與欲使用的@pdf得出mle variace-covariance矩陣
[p,k]=size(data);%測得data矩陣大小並將data分為當期與落後期
data1=data;
data1(1)=[];
data2=data;
data2(p)=[];
%套入matlab的mlecov語法
[q,nparams]=size(params);
ej = zeros(size(params));
ek = zeros(size(params));
nll = feval(pdf,data1,data2,params);
lnll=sum(log(nll));
nH = zeros(nparams,nparams);
for j = 1:nparams
    ej(j) = params(j)*0.01;
    for k = 1:(j-1)
        ek(k) = params(k)*0.01;
        % Four-point central difference for mixed second partials.
        a=feval(pdf,data1,data2,params+ej+ek);
        b=feval(pdf,data1,data2,params+ej-ek);
        c=feval(pdf,data1,data2,params-ej+ek);
        d=feval(pdf,data1,data2,params-ej-ek);
        nH(j,k) =sum(log(a)) ...
                -sum(log(b)) ...
                -sum(log(c)) ...
                +sum(log(d));
            
    end
    % Five-point central difference for pure second partial.
    e=feval(pdf,data1,data2,params+2*ej);
    f=16*feval(pdf,data1,data2,params+ej);
    g=16*feval(pdf,data1,data2,params-ej);
    h=feval(pdf,data1,data2,params-2*ej);
    nH(j,j) = -sum(log(e)) ...
            + sum(log(f)) ...
            - 30*lnll ...
            + sum(log(g)) ...
            - sum(log(h));
end
            % Fill in the upper triangle.
 nH = nH + triu(nH',1);
 nH = nH ./ (4.*(params(:)*0.01)*(params(:)*0.01)' + diag(8*(params(:)*0.01).^2));
 [R,p] = chol(nH);
if p > 0
    warning(message('stats:mlecov:NonPosDefHessian'));
    acov = NaN(nparams);
else
    % The asymptotic cov matrix approximation is the negative inverse of the
    % Hessian.
    Rinv = inv(R);
    acov = Rinv*Rinv';
end
