library(tree)
library(ISLR)
library(e1071)
library(class)
library(ggplot2)

projectdata=read.csv(file="C:/Users/james_000/Documents/Advanced Data Mining/NBAdataRF.csv",header=TRUE)
n=dim(projectdata)[1]
p=dim(projectdata)[2]-1
randproject=projectdata[sample(1:nrow(projectdata),474),]


X=randproject[,-1]
Y=randproject[,1]

n_train1=floor(p/2)
n_test1=n-n_train1
Train1=randproject[1:n_train1,]
Test1=randproject[(n_train1+1):n,]
n_train2=p
n_test2=n-n_train2
Train2=randproject[1:n_train2,]
Test2=randproject[(n_train2+1):n,]
n_train3=2*p
n_test3=n-n_train3
Train3=randproject[1:n_train3,]
Test3=randproject[(n_train3+1):n,]


#RF#
library(randomForest)
set.seed(1)
rfproj1=randomForest(x=Train1[,-1],y=Train1[,1],mtry=floor(sqrt(p)),importance=TRUE)
min(rfproj1$err.rate[,1])
POS.RF1=predict(rfproj1,newdata = Test1[,-1],type="class")
ErrorPOSRF1=table(POS.RF1,Test1[,1])
ErrorPOSRF1
ErrorPOSRFRATE1=1-sum(diag(ErrorPOSRF1))/nrow(Test1)
ErrorPOSRFRATE1
#rfproj1$err.rate
plot(x=1:500,y=rfproj1$err.rate[,1])
which.min(rfproj1$err.rate[,1])
importance(rfproj1)

rfproj2=randomForest(x=Train2[,-1],y=Train2[,1],mtry=floor(sqrt(p)),importance=TRUE)
min(rfproj2$err.rate[,1])
POS.RF2=predict(rfproj2,newdata = Test2[,-1],type="class")
ErrorPOSRF2=table(POS.RF2,Test2[,1])
ErrorPOSRF2
ErrorPOSRFRATE2=1-sum(diag(ErrorPOSRF2))/nrow(Test2)
ErrorPOSRFRATE2
plot(x=1:500,y=rfproj2$err.rate[,1])
which.min(rfproj2$err.rate[,1])
#importance(rfproj2)

rfproj3=randomForest(x=Train3[,-1],y=Train3[,1],mtry=floor(sqrt(p)),importance=TRUE)
min(rfproj3$err.rate[,1])
POS.RF3=predict(rfproj3,newdata = Test3[,-1],type="class")
ErrorPOSRF3=table(POS.RF3,Test3[,1])
ErrorPOSRF3
ErrorPOSRFRATE3=1-sum(diag(ErrorPOSRF3))/nrow(Test3)
ErrorPOSRFRATE3
plot(x=1:500,y=rfproj3$err.rate[,1])
which.min(rfproj3$err.rate[,1])
importance(rfproj3)

Cost=c(0.001,0.01,0.1,1,5,10,100)
Gamma=c(.1,.2,.3,.4,.5,.6,.7,.8,.9,1)
D=c(2,3,4,5,6,7,8,9,10,20,30,40,50,100)



#SVMlin#
svmlin1=svm(x=Train1[,-1],y=Train1[,1],scale = FALSE,type = "C-classification", kernel = "linear")
summary(svmlin1)
svmlin1.pred=predict(svmlin1,Test1[-1])
Errorsvmlin1=table(svmlin1.pred,Test1[,1])
Errorsvmlin1
Errorsvmlinrate1=1-sum(diag(Errorsvmlin1))/nrow(Test1)
Errorsvmlinrate1
set.seed(1)
#tune.svmlin1=tune.svm(x=Train1[,-1],y=Train1[,1],scale=FALSE,type = "C-classification",data = Train1,kernal="linear",cost = Cost)
tune.svmlin1=tune(svm,train.x = Train1[,-1],train.y = Train1[,1],scale = FALSE,type = "C-classification",ranges = list(cost=Cost))
summary(tune.svmlin1)
#bestmodsvmlin1=tune.svmlin1$best.model
bestmodsvmlin1
svmlin1bf=svm(x=Train1[,-1],y=Train1[,1],scale=FALSE,type = "C-classification", kernel = "linear",cost = bestmodsvmlin1$cost)
predsvmlin1=predict(svmlin1bf,Test1[,-1])
Errorsvmlinbf1=table(predsvmlin1,Test1[,1])
Errorsvmlinbf1
Errorsvmlinbfrate1=1-sum(diag(Errorsvmlinbf1))/nrow(Test1)
Errorsvmlinbfrate1

svmlin2=svm(x=Train2[,-1],y=Train2[,1],scale = FALSE,type = "C-classification", kernel = "linear")
summary(svmlin2)
svmlin2.pred=predict(svmlin2,Test2[-1])
Errorsvmlin2=table(svmlin2.pred,Test2[,1])
Errorsvmlin2
Errorsvmlinrate2=1-sum(diag(Errorsvmlin2))/nrow(Test2)
Errorsvmlinrate2
set.seed(2)
tune.svmlin2=tune(svm,train.x = Train2[,-1],train.y = Train2[,1],validation.x = Test2[,-1],validation.y = Test2[,1],scale = FALSE,type = "C-classification")
#summary(tune.svmlin2)
bestmodsvmlin2=tune.svmlin2$best.model
bestmodsvmlin2
svmlin2bf=svm(x=Train2[,-1],y=Train2[,1],scale=FALSE,type = "C-classification", kernel = "linear",cost = bestmodsvmlin2$cost)
predsvmlin2=predict(svmlin2bf,Test2[,-1])
Errorsvmlinbf2=table(predsvmlin2,Test2[,1])
Errorsvmlinbf2
Errorsvmlinbfrate2=1-sum(diag(Errorsvmlinbf2))/nrow(Test2)
Errorsvmlinbfrate2

svmlin3=svm(x=Train3[,-1],y=Train3[,1],scale = FALSE,type = "C-classification", kernel = "linear")
summary(svmlin3)
svmlin3.pred=predict(svmlin3,Test3[-1])
Errorsvmlin3=table(svmlin3.pred,Test3[,1])
Errorsvmlin3
Errorsvmlinrate3=1-sum(diag(Errorsvmlin3))/nrow(Test3)
Errorsvmlinrate3
set.seed(3)
tune.svmlin3=tune(svm,train.x = Train3[,-1],train.y = Train3[,1],validation.x = Test3[,-1],validation.y = Test3[,1],scale = FALSE,type = "C-classification")
#summary(tune.svmlin3)
bestmodsvmlin3=tune.svmlin3$best.model
bestmodsvmlin3
svmlin3bf=svm(x=Train3[,-1],y=Train3[,1],scale=FALSE,type = "C-classification", kernel = "linear",cost = bestmodsvmlin3$cost)
predsvmlin3=predict(svmlin3bf,Test3[,-1])
Errorsvmlinbf3=table(predsvmlin3,Test3[,1])
Errorsvmlinbf3
Errorsvmlinbfrate3=1-sum(diag(Errorsvmlinbf3))/nrow(Test3)
Errorsvmlinbfrate3

#poly
svmpoly1=svm(x=Train1[,-1],y=Train1[,1],scale = FALSE,type = "C-classification", kernel = "polynomial")
summary(svmpoly1)
svmpoly1.pred=predict(svmpoly1,Test1[-1])
Errorsvmpoly1=table(svmpoly1.pred,Test1[,1])
Errorsvmpoly1
Errorsvmpolyrate1=1-sum(diag(Errorsvmpoly1))/nrow(Test1)
Errorsvmpolyrate1
set.seed(1)
tune.svmpoly1=tune(svm,train.x = Train1[,-1],train.y = Train1[,1],scale = FALSE,type = "C-classification",ranges = list(Cost=Cost,Degree=D))
summary(tune.svmpoly1)
bestmodsvmpoly1=tune.svmpoly1$best.model
bestmodsvmpoly1
svmpoly1bf=svm(x=Train1[,-1],y=Train1[,1],scale=FALSE,type = "C-classification", kernel = "polynomial",degree=bestmodsvmpoly1$degree,cost = bestmodsvmpoly1$cost)
predsvmpoly1=predict(svmpoly1bf,Test1[,-1])
Errorsvmpolybf1=table(predsvmpoly1,Test1[,1])
Errorsvmpolybf1
Errorsvmpolybfrate1=1-sum(diag(Errorsvmpolybf1))/nrow(Test1)
Errorsvmpolybfrate1

svmpoly2=svm(x=Train2[,-1],y=Train2[,1],scale = FALSE,type = "C-classification", kernel = "polynomial")
summary(svmpoly2)
svmpoly2.pred=predict(svmpoly2,Test2[-1])
Errorsvmpoly2=table(svmpoly2.pred,Test2[,1])
Errorsvmpoly2
Errorsvmpolyrate2=1-sum(diag(Errorsvmpoly2))/nrow(Test2)
Errorsvmpolyrate2
set.seed(2)
tune.svmpoly2=tune(svm,train.x = Train2[,-1],train.y = Train2[,1],validation.x = Test2[,-1],validation.y = Test2[,1],scale = FALSE,type = "C-classification")
bestmodsvmpoly2=tune.svmpoly2$best.model
bestmodsvmpoly2
svmpoly2bf=svm(x=Train2[,-1],y=Train2[,1],scale=FALSE,type = "C-classification", kernel = "polynomial",degree=bestmodsvmpoly2$degree,cost = bestmodsvmpoly2$cost)
predsvmpoly2=predict(svmpoly2bf,Test2[,-1])
Errorsvmpolybf2=table(predsvmpoly2,Test2[,1])
Errorsvmpolybf2
Errorsvmpolybfrate2=1-sum(diag(Errorsvmpolybf2))/nrow(Test2)
Errorsvmpolybfrate2

svmpoly3=svm(x=Train3[,-1],y=Train3[,1],scale = FALSE,type = "C-classification", kernel = "polynomial")
summary(svmpoly3)
svmpoly3.pred=predict(svmpoly3,Test3[-1])
Errorsvmpoly3=table(svmpoly3.pred,Test3[,1])
Errorsvmpoly3
Errorsvmpolyrate3=1-sum(diag(Errorsvmpoly3))/nrow(Test3)
Errorsvmpolyrate3
set.seed(3)
tune.svmpoly3=tune(svm,train.x = Train3[,-1],train.y = Train3[,1],validation.x = Test3[,-1],validation.y = Test3[,1],scale = FALSE,type = "C-classification")
bestmodsvmpoly3=tune.svmpoly3$best.model
bestmodsvmpoly3
svmpoly3bf=svm(x=Train3[,-1],y=Train3[,1],scale=FALSE,type = "C-classification", kernel = "polynomial",degree=bestmodsvmpoly3$degree,cost = bestmodsvmpoly3$cost)
predsvmpoly3=predict(svmpoly3bf,Test3[,-1])
Errorsvmpolybf3=table(predsvmpoly3,Test3[,1])
Errorsvmpolybf3
Errorsvmpolybfrate3=1-sum(diag(Errorsvmpolybf3))/nrow(Test3)
Errorsvmpolybfrate3

#rad
svmrad1=svm(x=Train1[,-1],y=Train1[,1],scale = FALSE,type = "C-classification", kernel = "radial")
summary(svmrad1)
svmrad1.pred=predict(svmrad1,Test1[-1])
Errorsvmrad1=table(svmrad1.pred,Test1[,1])
Errorsvmrad1
Errorsvmradrate1=1-sum(diag(Errorsvmrad1))/nrow(Test1)
Errorsvmradrate1
set.seed(1)
tune.svmrad1=tune.svm(x=Train1[,-1],y=Train1[,1],scale=FALSE,data = Train1,kernal="radial",cost = Cost,gamma = Gamma)
bestmodsvmrad1=tune.svmrad1$best.model
bestmodsvmrad1
svmrad1bf=svm(x=Train1[,-1],y=Train1[,1],scale=FALSE,type = "C-classification", kernel = "radial",cost = bestmodsvmrad1$cost,gamma = bestmodsvmrad1$gamma)
predsvmrad1=predict(svmrad1bf,Test1[,-1])
Errorsvmradbf1=table(predsvmrad1,Test1[,1])
Errorsvmradbf1
Errorsvmradbfrate1=1-sum(diag(Errorsvmradbf1))/nrow(Test1)
Errorsvmradbfrate1

svmrad2=svm(x=Train2[,-1],y=Train2[,1],scale = FALSE,type = "C-classification", kernel = "radial")
summary(svmrad2)
svmrad2.pred=predict(svmrad2,Test2[-1])
Errorsvmrad2=table(svmrad2.pred,Test2[,1])
Errorsvmrad2
Errorsvmradrate2=1-sum(diag(Errorsvmrad2))/nrow(Test2)
Errorsvmradrate2
set.seed(2)
tune.svmrad2=tune.svm(x=Train2[,-1],y=Train2[,1],scale=FALSE,data = Train2,kernal="radial",cost = Cost,gamma = Gamma)
bestmodsvmrad2=tune.svmrad2$best.model
bestmodsvmrad2
svmrad2bf=svm(x=Train2[,-1],y=Train2[,1],scale=FALSE,type = "C-classification", kernel = "radial",cost = bestmodsvmrad2$cost,gamma = bestmodsvmrad2$gamma)
predsvmrad2=predict(svmrad2bf,Test2[,-1])
Errorsvmradbf2=table(predsvmrad2,Test2[,1])
Errorsvmradbf2
Errorsvmradbfrate2=1-sum(diag(Errorsvmradbf2))/nrow(Test2)
Errorsvmradbfrate2

svmrad3=svm(x=Train3[,-1],y=Train3[,1],scale = FALSE,type = "C-classification", kernel = "radial")
summary(svmrad3)
svmrad3.pred=predict(svmrad3,Test3[-1])
Errorsvmrad3=table(svmrad3.pred,Test3[,1])
Errorsvmrad3
Errorsvmradrate3=1-sum(diag(Errorsvmrad3))/nrow(Test3)
Errorsvmradrate3
set.seed(3)
tune.svmrad3=tune(svm,train.x = Train3[,-1],train.y = Train3[,1],validation.x = Test3[,-1],validation.y = Test3[,1],scale = FALSE,type = "C-classification")
bestmodsvmrad3=tune.svmrad3$best.model
bestmodsvmrad3
svmrad3bf=svm(x=Train3[,-1],y=Train3[,1],scale=FALSE,type = "C-classification", kernel = "radial",cost = bestmodsvmrad3$cost,gamma = bestmodsvmrad3$gamma)
predsvmrad3=predict(svmrad3bf,Test3[,-1])
Errorsvmradbf3=table(predsvmrad3,Test3[,1])
Errorsvmradbf3
Errorsvmradbfrate3=1-sum(diag(Errorsvmradbf3))/nrow(Test3)
Errorsvmradbfrate3

tune.svmlin1$best.parameters
tune.svmlin2$best.parameters
tune.svmlin3$best.parameters
tune.svmpoly1$best.parameters
tune.svmpoly2$best.parameters
tune.svmpoly3$best.parameters
tune.svmrad1$best.parameters
tune.svmrad2$best.parameters
tune.svmrad3$best.parameters