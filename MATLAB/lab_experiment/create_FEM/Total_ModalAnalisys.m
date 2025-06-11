clear;
clc;
CreateBeamElementParameters;
CreateTowerElementParameters;
CreateCableElementParameters;
MGlobalStiffnessAssemble;
MGlobalMassAssemble;
ModalAnalysis;
PlotModalshape;
%编写：简旭东、钱昆  2016.08于同济大学桥梁馆
%修改：简旭东        2018.05
%同济大学/桥梁工程系/桥梁振动控制与健康监测研究室