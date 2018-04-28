load('cluster_database');
M=nanmean(Age);
Age(isnan(Age))=M;
Embarked(62,1)=Embarked(61,1);
Embarked(830,1)=Embarked(829,1);
Sex=Sex-1;
sample=[Survived Sex Age SibSp Parch Fare Pclass];
[m,n]=size(sample);  
for i=1:n
%     sample(:,i)=sample(:,i)/norm(sample(:,i));
    sample(:,i)=(sample(:,i)-min(sample(:,i)))/(max(sample(:,i))-min(sample(:,i)));
end
sample1=[sample(:,3) sample(:,6)];
[Idx,C]=kmeans(sample,2);
 for i=1:891
   if Idx(i,1)==1 
      plot(sample1(i,1),sample1(i,2),'r*') % ��ʾ��һ��
     hold on 
   else if Idx(i,1)==2
         plot(sample1(i,1),sample1(i,2),'b*') %��ʾ�ڶ��� 
          hold on 
       else if Idx(i,1)==3 
             plot(sample1(i,1),sample1(i,2),'g*') %��ʾ������ 
             hold on 
          else if Idx(i,1)==4
                 plot(sample1(i,1),sample1(i,2),'k*') %��ʾ������
                  hold on 
           end 
       end 
    end 
  end 
 end 
