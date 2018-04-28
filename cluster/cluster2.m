load('cluster_database');
M=nanmean(Age);
Age(isnan(Age))=M;
Embarked(62,1)=Embarked(61,1);
Embarked(830,1)=Embarked(829,1);
sample=[Survived Pclass Sex Age SibSp Parch Fare];
[m,n]=size(sample);  
for i=1:n
    sample(:,i)=(sample(:,i)-min(sample(:,i)))/(max(sample(:,i))-min(sample(:,i)));
end
%取iris数据集中的两列，以便于画出聚类树
%sample=sample(:,1:2);
%figure(1);
%plot(sample(:,1),sample(:,2),'b*')
dis=pdist(sample);
distan=squareform(dis);%distan为距离矩阵
s=linkage(distan)%产生层次聚类的树结构
%figure(2);
dendrogram(s);