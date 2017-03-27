clear all
close all

%% load input data
load CollectedDatawithControl
dt=0.005;
Y_OUT=[];
X0_IN=[];%Control command
X1_IN=[];%X
X2_IN=[];%XDOT
for i=1:4
    Y_OUT=[Y_OUT;(XDOT_FIVESEC(2:end,i)-XDOT_FIVESEC(1:end-1,i))/dt];
    X0_IN=[X0_IN;CMD_FIVESEC(1:end-1,i)];
    X1_IN=[X1_IN;X_FIVESEC(1:end-1,i)];
    X2_IN=[X2_IN;XDOT_FIVESEC(1:end-1,i)];
end
X_IN=[X1_IN X2_IN];
Y_TARGET=Y_OUT-X0_IN;

LR=0.2;%Learning rate

[Theta1, Theta2]=randInitializeWeights(2,16,1);
J=zeros(1000,1);
[J(1) Theta1_grad Theta2_grad]=nnCostFunction(Theta1, Theta2, X_IN, Y_TARGET, 0);
for n=2:1000
    Theta1=Theta1-LR*Theta1_grad;
    Theta2=Theta2-LR*Theta2_grad;
    [J(n) Theta1_grad Theta2_grad]=nnCostFunction(Theta1, Theta2, X_IN, Y_TARGET, 0);
end

%inputs = houseInputs;
%targets = houseTargets;
 
% Create a Fitting Network
%hiddenLayerSize = 10;
%net = fitnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
%net.divideParam.trainRatio = 70/100;