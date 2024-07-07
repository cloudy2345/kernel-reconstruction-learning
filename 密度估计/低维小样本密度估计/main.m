function A=main(nchangshi)
a=-ones(nchangshi,1);
b=a;
zz = 0;
for k=1:nchangshi
    try
        [x_train,x_test,x_node,fz]=shuju5();  %ÃÜ¶È¹À¼Æ
        moxing=density3(x_train,x_node)
        n_test=length(x_test);
        fg1=zeros(n_test,1); 
        for i=1:n_test
            fg1(i)=moxing(x_test(i,:));
        end

       % fg2 = ksdensity(x_train,x_test);  %'Bandwidth',0.2
        fg2=ones(n_test,1);
        fzd=zeros(n_test,1);
        for i=1:n_test
            fzd(i)=fz(x_test(i,:));
        end
        recon=sum(log(fzd)-log(fg1+0.0000000000000000000001))/n_test;
        ksd=sum(log(fzd)-log(fg2+0.00000000000000000000001))/n_test;
      %  recon=sum((fzd-fg1).^2)/n_test;
      %  ksd=sum((fzd-fg2).^2)/n_test;
       
        %if l3>1.5*l1
        %    error('buyao')
        %end
        a(k)=recon;
        b(k)=ksd;

    catch
        zz=zz+1
    end
end
cishu=nchangshi-zz;
a=a(a~=-1,:);   %recon
b=b(b~=-1,:);   %ksd
a1=mean(a);
a2=std(a);
b1=mean(b);
b2=std(b);
A=[a1,a2,b1,b2,cishu];


end
