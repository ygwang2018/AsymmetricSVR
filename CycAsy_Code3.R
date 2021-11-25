### The asymmetric frameworks: an asymmetric SVR and an asymmetric LAV regression.

##  This is the source code of the simulation results in the manuscript "Optimal 

## electric load scheduling under asymmetric loss" by Mr Jinran Wu and Prof. You-Gan Wang

## School of Mathematical Sciences, Queensland University of Technology, Brisbane 4001, Australia

## Email: jinran.wu@hdr.qut.edu.au (J. Wu) and you-gan.wang@qut.edu.au (Y.-G???, Wang).

## In the simulation, New South Wales half-hourly electric load dataset is applied to test our

## proposed asymmetric frameworks.

## import the NSW electric load data in Feburary, 2019

DataSet1<-read.csv("/home/n10141065/AsyCase/201902_NSW.csv")

## import the NSW electric load data in March, 2019

DataSet2<-read.csv("/home/n10141065/AsyCase/201903_NSW.csv")


MFC<-function(ERRS, k1, k){
  
  k2<-k1*k
  
  PerCost<-ifelse(ERRS>0,k1*ERRS,k2*abs(ERRS))
  
  return(mean(PerCost)*48)
  
}


#According to Magnano and Boland (Energy, 2007), the Fourier transform is employer to detect the cycle pattern

#########Parameter Setting##############


######################################
CyclVariabs<-function(a){
  
  t<-c(1:a)
  
  SC<-sin(2*pi/48*t)
  
  CC<-cos(2*pi/48*t)
  
  X_Cycle<-list(SC=SC,CC=CC)
  
}


## the training data: Feburary, 2019

TrainSet<-DataSet1$TOTALDEMAND

TrainSize<-length(TrainSet)

## the test data: March 2019

TestSet<-DataSet2$TOTALDEMAND

TestSize<-length(TestSet)

A<-CyclVariabs(TrainSize+TestSize)


XTrainCycle<-cbind(A$SC[51:1344],A$CC[51:1344])

XTestCycle<-cbind(A$SC[1395:2256],A$CC[1395:2256])

##Prepare for trainning

InputTrain<-array()

for (i in c(1:48)){
  
  InputTrain<-cbind(InputTrain, TrainSet[i:(TrainSize-49+i)])
  
}

InputTrain<-InputTrain[1:1294,2:49]

OutputTrain<-TrainSet[51:TrainSize]

##Prepare for test

InputTest<-array()

for (i in c(1:48)){
  
  InputTest<-cbind(InputTest, TestSet[i:(TestSize-49+i)])
  
}

InputTest<-InputTest[1:862,2:49]

OutputTest<-TestSet[51:TestSize]

#######################################
## epsilon-support vector regression###
#######################################

library(e1071)

SVRInputTrain<-cbind(XTrainCycle,InputTrain)

SVRInputTest<-cbind(XTestCycle, InputTest)


SVR_start_time <- Sys.time()
Basic_SVR<-svm(SVRInputTrain, OutputTrain, type = "eps-regression",scale = TRUE,kernel="linear")

Basic_SVR_Prediction<-predict(Basic_SVR, SVRInputTest)
SVR_end_time <- Sys.time()

SVR_time=SVR_end_time - SVR_start_time

Basic_SVR_Error<-Basic_SVR_Prediction-OutputTest

mean(abs(Basic_SVR_Error)/OutputTest)

## LAV regression

library(L1pack)

LAV_TrainSet<-as.data.frame(cbind(XTrainCycle,InputTrain,OutputTrain))

LAV_start_time <- Sys.time()
Basic_LAV<-lad(formula = OutputTrain ~ ., data = LAV_TrainSet, method = "BR")

Basic_LAV_Prediction<-predict(Basic_LAV, newdata=as.data.frame(cbind(XTestCycle, InputTest)))
LAV_end_time <- Sys.time()
LAV_time<-LAV_end_time-LAV_start_time 

Basic_LAV_Error<-Basic_LAV_Prediction-OutputTest

mean(abs(Basic_LAV_Error)/OutputTest)

## Least sqaure regression
LS_TrainSet<-as.data.frame(cbind(XTrainCycle,InputTrain,OutputTrain))

LS_start_time <- Sys.time()
Basic_LS<-lm(formula = OutputTrain ~ ., data = LS_TrainSet)

Basic_LS_Prediction<-predict(Basic_LS, newdata=as.data.frame(cbind(XTestCycle, InputTest)))
LS_end_time <- Sys.time()
LS_time<-LS_end_time-LS_start_time 
Basic_LS_Error<-Basic_LS_Prediction-OutputTest

mean(abs(Basic_LS_Error)/OutputTest)


#########################################
## Multilayer perception#################
#########################################

library(neuralnet)

maxs <- apply(LAV_TrainSet, 2, max)

mins <- apply(LAV_TrainSet, 2, min)

Scale_TrainSet <- as.data.frame(scale(LAV_TrainSet,center = mins, scale = maxs - mins))

Scale_TestSet <- as.data.frame(scale(as.data.frame(cbind(XTestCycle,InputTest,OutputTest)),center = mins, scale = maxs - mins))

MLP_start_time <- Sys.time()
Basic_MLP<-neuralnet(formula = OutputTrain ~ .,Scale_TrainSet, hidden = 20, err.fct="sse",linear.output=TRUE,act.fct = "logistic")

Basic_MLP_Prediction <- predict(Basic_MLP, Scale_TestSet[,1:50])*(maxs[51]-mins[51])+mins[51]
MLP_end_time <- Sys.time()
MLP_time<-MLP_end_time-MLP_start_time 
Basic_MLP_Error<-Basic_MLP_Prediction-OutputTest

mean(abs(Basic_MLP_Error)/OutputTest)

###########################################################################################
################Asymmetric frameworks#######################################################
############################################################################################

############Data preprocessing#######

XYTrainSet<-as.data.frame(cbind(InputTrain,OutputTrain))

XYTestSet<-as.data.frame(cbind(InputTest,OutputTest))

means <- apply(XYTrainSet, 2, mean)

sds <- apply(XYTrainSet, 2, sd)

NScale_TrainSet <- as.data.frame(cbind(XTrainCycle,(XYTrainSet-means)/sds))

NScale_TestSet<-as.data.frame(cbind(XTestCycle,(XYTestSet-means)/sds))

###############################################
## Asymmetric LAV regression model training####
###############################################
AsyLAVStructure<-function(beta, TrainX, TrainY, k){
  
  beta<-array(beta)
  
  TrainX_extra=rep(1, times=dim(TrainX)[1])
  
  TrainX=cbind(TrainX_extra,TrainX)
  
  Xmatrix<-as.matrix(TrainX)
  
  Err<-Xmatrix %*% beta-TrainY
  
  SampleLLC<-ifelse(Err<0,k*abs(Err),Err)
  
  TrainLLC<-sum(SampleLLC)
  
}
############################################################
k1=80/2

Optimal_C<-c(100,10,1000,100,1,1000,1)

Optimal_eps<-c(0.001,0.0001,0,0.001,0.0001,0.0001,0.001)

k2_all<-c(4:10)*100

Results<-array()

B<-array()

for (i in c(1:7)){
  
  k2=k2_all[i]/2
  
  k=k2/k1
  
  C=Optimal_C[i]
  
  eps=Optimal_eps[i]
  ### NScale_TrainSet
  
  #####initialize the parameter using lav regression####
  AsyLAV_start_time <- Sys.time()
  Scale_LAV<-lad(formula = OutputTrain ~ ., data = NScale_TrainSet , method = "BR")
  
  Scale_Low_Limt<-rep(-1,51)
  
  Scale_Upp_Limt<-rep(3,51)
  
  Asy_LAV_Coef<-optim(as.array(Scale_LAV$coefficients), function(beta) 
    AsyLAVStructure(beta, NScale_TrainSet[,1:50], NScale_TrainSet[,51], k),
    method="L-BFGS-B",lower=Scale_Low_Limt, upper = Scale_Upp_Limt)$par
  
  ### NScale_TestSet
  
  Scale_Asy_LAV_Prediction<-as.matrix(cbind(rep(1, dim(NScale_TestSet)[1]),
                                            NScale_TestSet[,1:50]))%*% as.array(Asy_LAV_Coef)
  
  Asy_LAV_Prediction<-Scale_Asy_LAV_Prediction*sds[49]+means[49]
  
  
  AsyLAV_end_time <- Sys.time()
  AsyLAV_time<-AsyLAV_end_time-AsyLAV_start_time 
  
  Asy_LAV_Error<-Asy_LAV_Prediction-OutputTest
  
  mean(abs(Asy_LAV_Error)/OutputTest)
  
  ####################################################
  ###### Asymmetric SVR regression model training#####
  ####################################################
  
  AsySVRStructure<-function(beta, TrainX, TrainY, k, C, eps)
    
  {
    
    beta<-array(beta)
    
    TrainX_extra=rep(1, times=dim(TrainX)[1])
    
    TrainX=cbind(TrainX_extra,TrainX)
    
    Xmatrix<-as.matrix(TrainX)
    
    Err<-Xmatrix %*% beta-TrainY
    
    InsenErr<-ifelse(abs(Err)>eps, Err, 0)
    
    SampleLLC<-ifelse(InsenErr<0,k*abs(InsenErr),InsenErr)
    
    L2<-sum((beta[2:dim(TrainX)[2]])^2)
    
    TrainLLC<- L2+C*sum(SampleLLC)
    
  }
  
  ############################################################################################
  AsySVR_start_time<-Sys.time()
  Asy_SVR_Coef<-optim(as.array(Scale_LAV$coefficients), function(beta) 
    AsySVRStructure(beta, NScale_TrainSet[,1:50], NScale_TrainSet[,51], k=k, C=C, eps=eps), 
    method="L-BFGS-B",lower=Scale_Low_Limt, upper = Scale_Upp_Limt)$par
  
  ### NScale_TestSet
  
  Scale_Asy_SVR_Prediction<-as.matrix(cbind(rep(1, dim(NScale_TestSet)[1]),
                                            NScale_TestSet[,1:50]))%*% as.array(Asy_SVR_Coef)
  
  Asy_SVR_Prediction<-Scale_Asy_SVR_Prediction*sds[49]+means[49]
  AsySVR_end_time <- Sys.time()
  AsySVR_time<-AsySVR_end_time-AsySVR_start_time 
  Asy_SVR_Error<-Asy_SVR_Prediction-OutputTest
  
  mean(abs(Asy_SVR_Error)/OutputTest)
  
  ####################Train###########
  Scale_Asy_SVR_Prediction_Train<-as.matrix(cbind(rep(1, dim(NScale_TrainSet[,1:50])[1]),
                                                  NScale_TrainSet[,1:50]))%*% as.array(Asy_SVR_Coef)
  
  Asy_SVR_Prediction_Train<-Scale_Asy_SVR_Prediction_Train*sds[49]+means[49]
  
  Asy_SVR_Error_Train<-Asy_SVR_Prediction_Train-OutputTrain
  
  Asy_SVR_Cost_Train<-MFC(Asy_SVR_Error_Train,k1=k1,k=k)
  
  Asy_SVR_Cost_Train
  
  ######Results Presetation################
  
  ##########PREDICTIONS###################
  
  #Basic_SVR_Prediction
  
  #Basic_LAV_Prediction
  
  #Basic_MLP_Prediction
  
  #Asy_LAV_Prediction
  
  #sy_SVR_Prediction
  
  ##########ERRORS########
  
  ##Setting: k1=80, k2=k*k1
  
  #Basic_LS_Error
  
  Basic_LS_Cost<-MFC(Basic_LS_Error,k1=k1,k=k)
  
  Basic_LS_Cost
  
  #Basic_LAV_Error
  
  Basic_LAV_Cost<-MFC(Basic_LAV_Error,k1=k1,k=k)
  
  Basic_LAV_Cost
  
  #Basic_SVR_Error
  
  Basic_SVR_Cost<-MFC(Basic_SVR_Error,k1=k1,k=k)
  
  Basic_SVR_Cost
  
  #Basic_MLP_Error
  
  Basic_MLP_Cost<-MFC(Basic_MLP_Error,k1=k1,k=k)
  
  Basic_MLP_Cost
  
  #Asy_LAV_Error
  
  Asy_LAV_Cost<-MFC(Asy_LAV_Error,k1=k1,k=k)
  
  Asy_LAV_Cost
  
  #Asy_SVR_Error
  
  Asy_SVR_Cost<-MFC(Asy_SVR_Error,k1=k1,k=k)
  
  Asy_SVR_Cost
  
  #############################################
  
  Prediction_Results<-cbind(OutputTest,Basic_LS_Prediction,Basic_LAV_Prediction,Basic_SVR_Prediction,
                            
                            Basic_MLP_Prediction,Asy_LAV_Prediction,Asy_SVR_Prediction)
  
  Error_Results<-cbind(Basic_LS_Error,Basic_LAV_Error,Basic_SVR_Error,
                       Basic_MLP_Error,Asy_LAV_Error, Asy_SVR_Error)
  
  B<-cbind(B,Error_Results)
  
  #Output_Results<-as.data.frame(cbind(Prediction_Results,Error_Results))
  
  #write.csv(Output_Results,file="Model_Output_NSW600.csv")
  
  LS_Ratio<-table(sign(Basic_LS_Error))[2]/length(Basic_LS_Error)
  
  LAV_Ratio<-table(sign(Basic_LAV_Error))[2]/length(Basic_LS_Error)
  
  SVR_Ratio<-table(sign(Basic_SVR_Error))[2]/length(Basic_LS_Error)
  
  MLP_Ratio<-table(sign(Basic_MLP_Error))[2]/length(Basic_LS_Error)
  
  Asy_LAV_Ratio<-table(sign(Asy_LAV_Error))[2]/length(Basic_LS_Error)
  
  Asy_SVR_Ratio<-table(sign(Asy_SVR_Error))[2]/length(Basic_LS_Error)
  
  Ratio_Results<-rbind(LS_Ratio,LAV_Ratio,SVR_Ratio,MLP_Ratio,Asy_LAV_Ratio,Asy_SVR_Ratio)
  
  Cost_Results<-rbind(Basic_LS_Cost, Basic_LAV_Cost, Basic_SVR_Cost,Basic_MLP_Cost, Asy_LAV_Cost, Asy_SVR_Cost)
  
  Time_Results<-rbind(LS_time, LAV_time, SVR_time, MLP_time, AsyLAV_time,AsySVR_time)
  
  A<-cbind(Cost_Results,Ratio_Results,Time_Results)
  
  Results<-rbind(Results,A)
  
}
write.csv(Results,"Model_Indexes3.csv")
write.csv(B,"Model_Errors3.csv")

###################Quantile regression#####

#Basic_LAV<-lad(formula = OutputTrain ~ ., data = LAV_TrainSet, method = "BR")

#Basic_LAV_Prediction<-predict(Basic_LAV, newdata=as.data.frame(cbind(XTestCycle, InputTest)))

library(quantreg)

k1<-80/2

for (j in c(600,700,1000))
{
  
  k2<-j/2
  
  k<-k2/k1
  
  tau1<-k/(1+k)
  
  fit<-rq(formula = OutputTrain ~ ., data = NScale_TrainSet , tau= tau1)
  
  Scale_QR_Prediction<-predict(fit, newdata = as.data.frame(NScale_TestSet[,1:50]))
  
  QR_Prediction<-Scale_QR_Prediction*sds[49]+means[49]
  
  QR_Error<-QR_Prediction-OutputTest
  
  TAU<-1-sum(sign(QR_Error<0))/length(Basic_LS_Error)
  
  fit_qr_Cost<-MFC(QR_Error,k1=k1,k=k)
  
  print(cbind(tau1,TAU,fit_qr_Cost))
  
}