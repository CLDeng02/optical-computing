%%
%-------------------------------光衍射神经网络实现二元逻辑运算--------------------------------------------------------
clc;
clear;
close all;
addpath(genpath('./function'));
N=4; % number of cascaded amplitude-phase masks (L=N-1); the last mask is conjugate(P)
size1=128;
size2=128;
%paramaters in simulating the Fresnel field propagation
commask=ones(size1+32,2*size2+48,N);%initial values for the cascaded amplitude-phase masks

g=6;%生成图像对数
g1=g-1;%训练图像对数
intp=zeros(size1+32,2*size2+48,g);%输入图像对
outp=zeros(size1+32,2*size2+48,g);%理论计算结果
%-------------------------------------------------------生成测试图----------------------------------------------------
%随机产生图像对
for iter_0=1:2*g
tran1=2*rand(16,16);
tran1(tran1<0.8)=0;
tran1(tran1>=0.8)=1;
pic1=imresize(tran1,[size1 size2],'nearest');
pic1(pic1<0.5)=0;
pic1(pic1>=0.5)=1;
pic(:,:,iter_0)=pic1;
%imshow(pic(:,:,iter_0))
%pause(0.01)
end

%与计算
for iter_1=1:g
    input1=pic(:,:,iter_1);
    input2=pic(:,:,iter_1+g);
  intp(17:16+size1,17:32+2*size2,iter_1)=[input1,zeros(size1,16),input2];
  for j=1:size1
    for j1=1:size2
      if input1(j,j1)==1 && input2(j,j1)==1
      outp1(j,j1)=1;
      else
      outp1(j,j1)=0;
      end
    end
  end
  output(:,:,iter_1)=outp1;
end
outp(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2),:)=output;
%%
p_num=1;%第几张计算图
subplot(2,1,1)
imshow(intp(:,:,p_num));title('输入');
subplot(2,1,2)
imshow(outp(:,:,p_num));title('与计算输出');
%%
%产生初始相位分布
[r,c]=size(commask(:,:,1));
for iter0=1:N
phi(:,:,iter0)=2*pi*rand(r,c);
end
commask=exp(1i*phi);
%%
%第一种更新方式，每层用多张图片进行更新，每张图片循环100次（但是多图片不收敛）
beta1=0.9;
beta2=0.999;
v=zeros(r,c);
s=zeros(r,c);
b=0.0001;
r1=0.1;%学习率
wavelen=532e-9;%波长m
dist=0.01;%衍射距离m
pixsize=4e-6;%像素尺寸m
step=1;

for epoch=1:1  %完整循环次数
for iter=1:N       %相位板循环
    phi0=phi(:,:,iter);
    
   for iter1=1:g1  %训练图循环
       
       intpp=intp(:,:,iter1);
       backp=output(:,:,iter1);
       
       intgp=intpp;
       for iter2=1:iter
          intwave=propagate(intgp,dist,pixsize,wavelen);
          int_wave=intwave;%提供需要更新的相位板处的波前函数
          intgp=intwave.*commask(:,:,iter2);
       end
       
       
   for j=1:100 %adm迭代次数
       
       intpg=intpp;
      for iterp=1:N  %正向衍射计算
          intwaveg=propagate(intpg,dist,pixsize,wavelen);
          intpg=intwaveg.*commask(:,:,iterp);
      end
      outwaveg=propagate(intpg,dist,pixsize,wavelen);%记录面复振幅
      
      A=outwaveg;
      ab=A.*conj(A);
      ab=ab./max(ab(:));
      ab_cut=ab(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2));
      Ae=ab_cut-backp;
      error(step)=sum(Ae(:).^2)./(r*c);
      dA=zeros(size1+32,2*size2+48);
      dA(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2))=Ae;
      backorg=dA.*A.*2;
      
      backg=backorg;
      for iter3=1:N-iter+1
          backwave=propagate(backg,-dist,pixsize,wavelen);%回传
          back_wave=backwave;
          backg=backwave.*conj(commask(:,:,N-iter3+1));
      end
      grd=real(conj(int_wave).*back_wave.*(-1i*conj(commask(:,:,iter))));
      
      %启用自适应adam优化
      v=beta1*v+(1-beta1)*grd;
      s=beta2*s+(1-beta2)*grd.^2;
      vc=v./(1-beta1^j);
      sc=s./(1-beta2^j);
      m=vc./(sqrt(sc)+b).*r1;
      phi0=phi0-m;
      
      commask(:,:,iter)=exp(1i*phi0);  %更新相位  
      step=step+1
   end
   end
   phi(:,:,iter)=phi0;
end
end
%%
%第二种更新方式，一张图片训练整个网络
beta1=0.9;
beta2=0.999;
v1=zeros(r,c,N);
s1=zeros(r,c,N);
j_num=ones(1,N);
b=0.0001;
r1=0.08;%学习率
wavelen=532e-9;%波长m
dist=0.01;%衍射距离m
pixsize=4e-6;%像素尺寸m
step=1;
err=100;


for epoch=1:60  %完整循环次数
    epoch
for iter1=1:g1  %训练图循环
    for iter=1:N       %相位板循环
       phi0=phi(:,:,iter);
    
       intpp=intp(:,:,iter1);
       backp=output(:,:,iter1);
       
       intgp=intpp;
       for iter2=1:iter
          intwave=propagate(intgp,dist,pixsize,wavelen);
          int_wave=intwave;%提供需要更新的相位板处的波前函数
          intgp=intwave.*commask(:,:,iter2);
       end
       
       intpg=intpp;
      for iterp=1:N  %正向衍射计算
          intwaveg=propagate(intpg,dist,pixsize,wavelen);
          intpg=intwaveg.*commask(:,:,iterp);
      end
      outwaveg=propagate(intpg,dist,pixsize,wavelen);%记录面复振幅
      
      A=outwaveg;
      ab=A.*conj(A);
      ab=ab./max(ab(:));
      ab_cut=ab(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2));
      Ae=ab_cut-backp;
      error(step)=sum(Ae(:).^2)./(r*c);
      err=error(step);
      dA=zeros(size1+32,2*size2+48);
      dA(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2))=Ae;
      backorg=dA.*A.*2;
      
      backg=backorg;
      for iter3=1:N-iter+1
          backwave=propagate(backg,-dist,pixsize,wavelen);%回传
          back_wave=backwave;
          backg=backwave.*conj(commask(:,:,N-iter3+1));
      end
      grd=real(conj(int_wave).*back_wave.*(-1i*conj(commask(:,:,iter))));
      
      
      %启用自适应adam优化
      v=v1(:,:,iter);
      s=s1(:,:,iter);
      v=beta1*v+(1-beta1)*grd;
      s=beta2*s+(1-beta2)*grd.^2;
      vc=v./(1-beta1^j_num(1,iter));
      sc=s./(1-beta2^j_num(1,iter));
      m=vc./(sqrt(sc)+b).*r1;
      phi0=phi0-m;
      
      
      commask(:,:,iter)=exp(1i*phi0);  %更新相位  
      step=step+1;
      phi(:,:,iter)=phi0;
      j_num(1,iter)=j_num(1,iter)+1;
      v1(:,:,iter)=v;
      s1(:,:,iter)=s;
   end
  
end
end
%%
%第三种更新方式，多张图片训练一层网络,从前往后更新
beta1=0.9;
beta2=0.999;
v1=zeros(r,c,N);
s1=zeros(r,c,N);
j_num=ones(1,N);
b=0.0001;
r1=0.1;%学习率
wavelen=532e-9;%波长m
dist=0.01;%衍射距离m
pixsize=4e-6;%像素尺寸m
step=1;

for epoch=1:100  %完整循环次数
    epoch
for iter=1:N       %相位板循环
       phi0=phi(:,:,iter);
    for iter1=1:g1  %训练图循环
       
    
       intpp=intp(:,:,iter1);
       backp=output(:,:,iter1);
       
       intgp=intpp;
       for iter2=1:iter
          intwave=propagate(intgp,dist,pixsize,wavelen);
          int_wave=intwave;%提供需要更新的相位板处的波前函数
          intgp=intwave.*commask(:,:,iter2);
       end
       
       intpg=intpp;
      for iterp=1:N  %正向衍射计算
          intwaveg=propagate(intpg,dist,pixsize,wavelen);
          intpg=intwaveg.*commask(:,:,iterp);
      end
      outwaveg=propagate(intpg,dist,pixsize,wavelen);%记录面复振幅
      
      A=outwaveg;
      ab=A.*conj(A);
      ab=ab./max(ab(:));
      ab_cut=ab(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2));
%       ab_cut(ab_cut<0.5)=0;
%       ab_cut(ab_cut>=0.5)=1;%二值化
      Ae=ab_cut-backp;
      error(step)=sum(Ae(:).^2)./(r*c);
      dA=zeros(size1+32,2*size2+48);
      dA(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2))=Ae;
      backorg=dA.*A.*2;
      
      backg=backorg;
      for iter3=1:N-iter+1
          backwave=propagate(backg,-dist,pixsize,wavelen);%回传
          back_wave=backwave;
          backg=backwave.*conj(commask(:,:,N-iter3+1));
      end
      grd=real(conj(int_wave).*back_wave.*(-1i*conj(commask(:,:,iter))));
      
      
      %启用自适应adam优化
      v=v1(:,:,iter);
      s=s1(:,:,iter);
      v=beta1*v+(1-beta1)*grd;
      s=beta2*s+(1-beta2)*grd.^2;
      vc=v./(1-beta1^j_num(1,iter));
      sc=s./(1-beta2^j_num(1,iter));
      m=vc./(sqrt(sc)+b).*r1;
      phi0=phi0-m;
      
      
      commask(:,:,iter)=exp(1i*phi0);  %更新相位  
      step=step+1;
      phi(:,:,iter)=phi0;
      j_num(1,iter)=j_num(1,iter)+1;
      v1(:,:,iter)=v;
      s1(:,:,iter)=s;
   end
  
end
end
%%
%第四种更新方式，一张图片训练多层网络，从后往前更新相位板
beta1=0.9;
beta2=0.999;
v1=zeros(r,c,N);
s1=zeros(r,c,N);
j_num=ones(1,N);
b=0.0001;
r1=0.01;%学习率
wavelen=532e-9;%波长m
dist=0.01;%衍射距离m
pixsize=4e-6;%像素尺寸m
step=1;


for epoch=1:30  %完整循环次数
    epoch
    
for iter1=1:g1  %训练图循环
    for iter_z=1:N       %相位板循环
        iter=N-iter_z+1;
       phi0=phi(:,:,iter);
     
       intpp=intp(:,:,iter1);
       backp=output(:,:,iter1);
       
       intgp=intpp;
       for iter2=1:iter
          intwave=propagate(intgp,dist,pixsize,wavelen);
          int_wave=intwave;%提供需要更新的相位板处的波前函数
          intgp=intwave.*commask(:,:,iter2);
       end
       
       intpg=intpp;
      for iterp=1:N  %正向衍射计算
          intwaveg=propagate(intpg,dist,pixsize,wavelen);
          intpg=intwaveg.*commask(:,:,iterp);
      end
      outwaveg=propagate(intpg,dist,pixsize,wavelen);%记录面复振幅
      
      A=outwaveg;
      ab=A.*conj(A);
      ab=ab./max(ab(:));
      ab_cut=ab(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2));
      Ae=ab_cut-backp;
      error(step)=sum(Ae(:).^2)./(r*c);
      err=error(step);
      dA=zeros(size1+32,2*size2+48);
      dA(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2))=Ae;
      backorg=dA.*A.*2;%./(r*c);
      
      backg=backorg;
      for iter3=1:N-iter+1
          backwave=propagate(backg,-dist,pixsize,wavelen);%回传
          back_wave=backwave;
          backg=backwave.*conj(commask(:,:,N-iter3+1));
      end
      grd=real(conj(int_wave).*back_wave.*(-1i*conj(commask(:,:,iter))));
      
      
      %启用自适应adam优化
      v=v1(:,:,iter);
      s=s1(:,:,iter);
      v=beta1*v+(1-beta1)*grd;
      s=beta2*s+(1-beta2)*grd.^2;
      vc=v./(1-beta1^j_num(1,iter));
      sc=s./(1-beta2^j_num(1,iter));
      m=vc./(sqrt(sc)+b).*r1;
      phi0=phi0-m;
      
      
      commask(:,:,iter)=exp(1i*phi0);  %更新相位  
      step=step+1;
      phi(:,:,iter)=phi0;
      j_num(1,iter)=j_num(1,iter)+1;
      v1(:,:,iter)=v;
      s1(:,:,iter)=s;
   end
  
end
end
%%
%第五种更新方式，多张图片训练一层网络，取平均梯度
beta1=0.9;
beta2=0.999;
v1=zeros(r,c,N);
s1=zeros(r,c,N);
j_num=ones(1,N);
b=0.0001;
r1=0.05;%学习率
wavelen=532e-9;%波长m
dist=0.01;%衍射距离m
pixsize=6e-6;%像素尺寸m
step=1;
for epoch=1:100  %完整循环次数
    epoch
for iter=1:N       %相位板循环
       phi0=phi(:,:,iter);
       grd=double(zeros(r,c));
    for iter1=1:g1  %批处理训练图
       
    
       intpp=intp(:,:,iter1);
       backp=output(:,:,iter1);
       
       intgp=intpp;
       for iter2=1:iter
          intwave=propagate(intgp,dist,pixsize,wavelen);
          int_wave=intwave;%提供需要更新的相位板处的波前函数
          intgp=intwave.*commask(:,:,iter2);
       end
       
       intpg=intpp;
      for iterp=1:N  %正向衍射计算
          intwaveg=propagate(intpg,dist,pixsize,wavelen);
          intpg=intwaveg.*commask(:,:,iterp);
      end
      outwaveg=propagate(intpg,dist,pixsize,wavelen);%记录面复振幅
      
      A=outwaveg;
      ab=A.*conj(A);
      ab=ab./max(ab(:));
      ab_cut=ab(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2));
%       ab_cut(ab_cut<0.5)=0;
%       ab_cut(ab_cut>=0.5)=1;%二值化
      Ae=ab_cut-backp;
      error(step)=sum(Ae(:).^2)./(r*c);
     
      dA=zeros(size1+32,2*size2+48);
      dA(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2))=Ae;
      backorg=dA.*A.*2;
      
      backg=backorg;
      for iter3=1:N-iter+1
          backwave=propagate(backg,-dist,pixsize,wavelen);%回传
          back_wave=backwave;
          backg=backwave.*conj(commask(:,:,N-iter3+1));
      end
      grdp=real(conj(int_wave).*back_wave.*(-1i*conj(commask(:,:,iter))));
      grdp=grdp./max(abs(grdp));
      grd=grd+grdp;

      clear intwave intwaveg backwave
   end
   
      grd=grd./g1;%除以批处理数
      step=step+1;
  
      %启用自适应adam优化
      v=v1(:,:,iter);
      s=s1(:,:,iter);
      v=beta1*v+(1-beta1)*grd;
      s=beta2*s+(1-beta2)*grd.^2;
      vc=v./(1-beta1^j_num(1,iter));
      sc=s./(1-beta2^j_num(1,iter));
      m=vc./(sqrt(sc)+b).*r1;
      phi0=phi0-m;
      mGrd(iter,epoch)=sum(m(:))./(r*c);
      
      commask(:,:,iter)=exp(1i*phi0);  %更新相位  
      
      phi(:,:,iter)=phi0;
      j_num(1,iter)=j_num(1,iter)+1;
      v1(:,:,iter)=v;
      s1(:,:,iter)=s;

      clear v s vc sc
end
end

%%
%误差绘制
%eror1=load('error1.mat', 'error');
[~,len]=size(error);
plot(1:len,error,'b-');
title('均方误差图');
xlabel('iteration')
ylabel('MSE')
% save('error.mat','error')
% save('mask','commask')
%%
%梯度检查
for i=1:N
subplot(4,1,i)
plot(1:100,mGrd(i,:),'b-')
axis([1,100,-5e-5,5e-5])
end
% save('mGrddata.mat','mGrd')
%%
%测试
ceshi=5;
    intce=intp(:,:,ceshi);%测试图有5对，从16开始到20
    int_ce=intce;
for ce=1:N  %正向衍射计算
    ce_wave=propagate(int_ce,dist,pixsize,wavelen);
    int_ce=ce_wave.*commask(:,:,ce);
end
    out_ce=propagate(int_ce,dist,pixsize,wavelen);
    out_ce=out_ce(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2));
    out1=outp(:,:,ceshi);
    out1=out1(17:16+size1,25+floor(size2/2):24+size2+floor(size2/2));
subplot(2,2,1)
imshow(intce),title('输入');
subplot(2,2,2)
imshow(abs(out_ce).^2,[]),title('模拟计算值');
subplot(2,2,4)
imshow(out1),title('理论计算值');
