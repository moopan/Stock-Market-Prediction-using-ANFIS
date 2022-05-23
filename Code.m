clear;clc;

%% init

yhat = [];
TrainingErrors = [];

%% load text data

T = readtable('GSPC-snp500.csv');

close_data = T{end-1050:end,5};
x = close_data';

%% the main for loop: iterate through dates
% 2 months test

TEST_DAY = 7; %42

for day = TEST_DAY:-1:1
    x = close_data(end-day-999:end-day+1)';

    %% prepare data

    % model 1, weekly data
    Delays = [5 10 15 20 25];
    [InputsW, TargetsW] = CreateTimeSeriesData(x, Delays);

    %% model2: daily data
    Delays = [1 2 3 4 5];
    [InputsD, TargetsD] = CreateTimeSeriesData(x, Delays);

    %% split train & test

    [XTrainW, XTestW, YTrainW, YTestW] = CreateTimedTrainTestData(InputsW', TargetsW');
    [XTrainD, XTestD, YTrainD, YTestD] = CreateTimedTrainTestData(InputsD', TargetsD');


    %% FIS Generation

    %%
    % 2: 'Subtractive Clustering (genfis2)';
    Radius=0.55;
    %fis2=genfis2(TrainInputs,TrainTargets,Radius);

    opt = genfisOptions('SubtractiveClustering',...
                        'ClusterInfluenceRange',Radius);
    fis2D = genfis(XTrainD, YTrainD, opt);
    fis2W = genfis(XTrainW, YTrainW, opt);
    %showrule(fis2)

    %% tune fis 2

    [in,out,~] = getTunableSettings(fis2D);
    opt2 = tunefisOptions("Method","anfis");
    fisout2D = tunefis(fis2D,[in;out],XTrainD,YTrainD,opt2);

    [in,out,rule2] = getTunableSettings(fis2W);
    opt2 = tunefisOptions("Method","anfis");
    fisout2W = tunefis(fis2W,[in;out],XTrainW,YTrainW,opt2);

    %% Train ANFIS
    MaxEpoch=50;
    ErrorGoal=0;
    InitialStepSize=0.01;
    StepSizeDecreaseRate=0.9;
    StepSizeIncreaseRate=1.1;
    TrainOptions=[MaxEpoch ...
                  ErrorGoal ...
                  InitialStepSize ...
                  StepSizeDecreaseRate ...
                  StepSizeIncreaseRate];
    DisplayInfo=true;
    DisplayError=true;
    DisplayStepSize=true;
    DisplayFinalResult=true;
    DisplayOptions=[DisplayInfo ...
                    DisplayError ...
                    DisplayStepSize ...
                    DisplayFinalResult];
    OptimizationMethod=1;
    % 0: Backpropagation
    % 1: Hybrid

    %% optmethod = 1
    %no tuning            
    
    anfis2d=anfis([XTrainD YTrainD],fis2D,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis2w=anfis([XTrainW YTrainW],fis2W,TrainOptions,DisplayOptions,[],OptimizationMethod);


    %tuned            
    anfis2dt=anfis([XTrainD YTrainD],fisout2D,TrainOptions,DisplayOptions,[],OptimizationMethod);
    anfis2wt=anfis([XTrainW YTrainW],fisout2W,TrainOptions,DisplayOptions,[],OptimizationMethod);

    %% Apply ANFIS to Data

    % daily models
    OutputsD = zeros(2,size(InputsD,2));

    OutputsD(1,:)=evalfis(InputsD,anfis2d);
    OutputsD(2,:)=evalfis(InputsD,anfis2dt);

    TrainOutputsD=OutputsD(:, 1:end-1);
    TestOutputsD=OutputsD(:, end);

    % weekly models
    OutputsW = zeros(2,size(InputsW,2));
    
    OutputsW(1,:)=evalfis(InputsW,anfis2w);
    OutputsW(2,:)=evalfis(InputsW,anfis2wt);

    TrainOutputsW=OutputsW(:, 1:end-1);
    TestOutputsW=OutputsW(:, end);

   
    nmin = min(size(TrainOutputsD,2),size(TrainOutputsW,2));
    TrainOutputsD = TrainOutputsD(2,end-nmin+1:end);
    TrainOutputsW = TrainOutputsW(2,end-nmin+1:end);
    
    TrainOutputs = [TrainOutputsD; TrainOutputsW];
    TestOutputs = [TestOutputsD; TestOutputsW];

    yhat = [yhat; TestOutputs' mean(TestOutputs)];
    
    TrainOutputs = mean(TrainOutputs)';
    TestOutputs = mean(TestOutputs)';

    %% Error Calculation
    nmin = min(size(YTrainD,1),size(YTrainW,1));
    TrainTargets = YTrainD(end-nmin+1:end,1);
    TestTargets = YTestD;

    TrainErrors=TrainTargets-TrainOutputs;
    TrainMSE=mean(TrainErrors.^2);
    TrainRMSE=sqrt(TrainMSE);
    TrainErrorMean=mean(TrainErrors);
    TrainErrorSTD=std(TrainErrors);
    TestErrors=TestTargets-TestOutputs;
    TestMSE=mean(TestErrors.^2);
    TestRMSE=sqrt(TestMSE);
    TestErrorMean=mean(TestErrors);
    TestErrorSTD=std(TestErrors);

    TrainingErrors = [TrainingErrors; TrainMSE TrainRMSE TrainErrorMean TrainErrorSTD];
end


%% Plot Results

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');