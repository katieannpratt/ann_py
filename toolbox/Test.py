from __future__ import division
import pandas as pd
import random
import numpy as np
import copy
from sklearn.metrics import roc_curve, auc

from toolbox import Data_Manipulation


class Test_Data(object):

	def __init__(self, parent, Testing_Data, Weights):
		self.Main_Menu = parent
		self.Testing_Data = Testing_Data
		self.Weights = Weights
		exec "self.Activation_Function = " + self.Main_Menu.Activation_Function
		self.Prep_Data()
		self.Add_Calculated_Targets()

	def Add_Calculated_Targets(self):
		Output_Data = []
		for Instance in self.Epoch:
			Outputs = Data_Manipulation.Feed_Forward(Instance[0], self.Weights, self.Activation_Function)
			Output_Data.append(Outputs[-1])
		self.Testing_Data['Output_Results'] = Output_Data
		if self.Main_Menu.Activation_Method == 'Hyperbolic Tangent':
			Function = lambda x: [-1 if i < 0 else i for i in x]
		else:
			Function = lambda x: [0 if i < .5 else i for i in x]
		self.Testing_Data['Modified_Results'] = map(Function, self.Testing_Data['Output_Results'])
		Function = lambda x: [1 if 0 < i == max(x) else (i if i <= 0 else 0) for i in x]
		self.Testing_Data['Modified_Results'] = map(Function, self.Testing_Data['Modified_Results'])

	def Prep_Data(self):
		self.Epoch = []
		for Index in self.Testing_Data.index:
			Inputs = []
			Attribute = 0
			while Attribute < self.Main_Menu.Attribute_Count:
				Inputs.append(self.Testing_Data[self.Testing_Data.columns[Attribute]][Index])
				Attribute += 1
			Target = self.Testing_Data[self.Testing_Data.columns[Attribute]][Index]
			self.Epoch.append([Inputs, Target])

	def Evaluate(self, DataFrame):
		self.Testing_Data = DataFrame
		self.Create_Confusion_Matrix()
		self.Calculate_Errors()
		self.Calculate_Correlation_Coefficient()
		self.Calculate_AUC()
		self.Calculate_Observed_Accuracy()
		self.Calculate_Kappa_Statistic()
		self.Build_DataFrame()

	def Create_Confusion_Matrix(self):
		if self.Main_Menu.Possible_Classes == [1]:
			Function = lambda x: x[0]
		else:
			Function = lambda x: 'Unmapped' if set(x) == set([0]) else 'Unmapped' if set(x) == set([-1]) else self.Main_Menu.Possible_Classes[[x.index(i) for i in x if i == 1][0]]
		#Add in missing values, if exists
		Predicted_Ys = map(Function, self.Testing_Data['Modified_Results'])
		Target_Ys = map(Function, self.Testing_Data['Target'])
		self.Confusion_DF =  pd.crosstab([Target_Ys], [Predicted_Ys], colnames=['Predicted'], rownames=['Actual'], dropna = False)
		if len(set(Predicted_Ys) - set(Target_Ys)) > 0:
			Add_Target_Set = set(Predicted_Ys) - set(Target_Ys)
			while Add_Target_Set:
				DF_Row = pd.DataFrame([np.nan], columns = [self.Confusion_DF.columns[1]])
				DF_Row = DF_Row.rename(index={0: Add_Target_Set.pop()})
				self.Confusion_DF = self.Confusion_DF.append(DF_Row)
		if len(set(Target_Ys) - set(Predicted_Ys)) > 0:
			Add_Predicted_Set = set(Target_Ys) - set(Predicted_Ys)
			while Add_Predicted_Set:
				self.Confusion_DF[Add_Predicted_Set.pop()] = np.nan
		self.Confusion_DF = self.Confusion_DF.fillna(0)
		self.Confusion_Norm_DF = copy.deepcopy(self.Confusion_DF)
		for i in self.Confusion_Norm_DF.columns:
			self.Confusion_Norm_DF[i] = self.Confusion_DF[i]/self.Confusion_DF.sum(axis=1)

	def Calculate_AUC(self):
		Index = 0
		self.ROC_AUC = []
		self.FPR = []
		self.TPR = []
		while Index < len(self.Main_Menu.Possible_Classes):
			Function = lambda x: x[Index]
			Target_Y = map(Function, self.Testing_Data['Target'])
			Predicted_Y = map(Function, self.Testing_Data['Output_Results'])
			False_Positive_Rate, True_Positive_Rate, Thresholds = roc_curve(Target_Y, Predicted_Y)
			self.ROC_AUC.append(auc(False_Positive_Rate, True_Positive_Rate))
			self.FPR.append(False_Positive_Rate)
			self.TPR.append(True_Positive_Rate)
			Index += 1

	def Calculate_Correlation_Coefficient(self):
		Index = 0
		self.Correlation_Coefficient = []
		while Index < len(self.Main_Menu.Possible_Classes):
			Function = lambda x: x[Index]
			Target_Y = map(Function, self.Testing_Data['Target'])
			Predicted_Y = map(Function, self.Testing_Data['Output_Results'])
			self.Correlation_Coefficient.append(np.corrcoef(Target_Y, Predicted_Y)[0][1])
			Index += 1

	def Calculate_Errors(self):
		Index = 0
		self.R_Squared = []
		self.Relative_Absolute_Error = []
		self.Mean_Absolute_Error = []
		self.Root_Mean_Squared_Error = []
		self.Root_Relative_Squared_Error = []
		while Index < len(self.Main_Menu.Possible_Classes):
			Residuals = lambda x, y: np.subtract(x,y)[Index]
			Mean = lambda x: sum([i[Index] for i in x])/float(len(x))
			Y_Mean = Mean(self.Testing_Data['Target'])
			Total_Instances = self.Testing_Data['Target'].count()
			Total_Squares = lambda x: x[Index] - Y_Mean
			RSS = sum([R * R for R in map(Residuals, self.Testing_Data['Target'], self.Testing_Data['Output_Results'])])
			RAV = sum([abs(R) for R in map(Residuals, self.Testing_Data['Target'], self.Testing_Data['Output_Results'])])
			TSS = sum([R * R for R in map(Total_Squares, self.Testing_Data['Target'])])
			TAV = sum([abs(R) for R in map(Total_Squares, self.Testing_Data['Target'])])
			self.R_Squared.append(1-(RSS/TSS))
			self.Relative_Absolute_Error.append(RAV/TAV)
			self.Mean_Absolute_Error.append(RAV/Total_Instances)
			self.Root_Mean_Squared_Error.append(np.sqrt(RSS/Total_Instances))
			self.Root_Relative_Squared_Error.append(np.sqrt(RSS/TSS))
			Index += 1
		Residuals, Total_Squares = self.Calculate_Euclidean_Distance()
		Total_Instances = self.Testing_Data['Target'].count()
		RSS = sum([R * R for R in Residuals])
		RAV = sum([abs(R) for R in Residuals])
		TSS = sum([R * R for R in Total_Squares])
		TAV = sum([abs(R) for R in Total_Squares])
		self.Overall_R_Squared = (1-(RSS/TSS))
		self.Overall_Relative_Absolute_Error = (RAV/TAV)
		self.Overall_Mean_Absolute_Error = (RAV/Total_Instances)
		self.Overall_Root_Mean_Squared_Error = (np.sqrt(RSS/Total_Instances))
		self.Overall_Root_Relative_Squared_Error = (np.sqrt(RSS/TSS))

	def Calculate_Euclidean_Distance(self):
		Residual_Function = lambda x, y: np.sqrt(sum(np.power((np.subtract(x,y)),2)))
		Means = []
		Index = 0
		while Index < len(self.Main_Menu.Possible_Classes):
			Mean = lambda x: sum([i[Index] for i in x])/float(len(x))
			Y_Mean = Mean(self.Testing_Data['Target'])
			Means.append(Y_Mean)
			Index += 1
		Total_Squares_Function = lambda x: np.sqrt(sum(np.power((np.subtract(x,Means)),2)))
		Residuals = map(Residual_Function, self.Testing_Data['Target'], self.Testing_Data['Output_Results'])
		Total_Squares = map(Total_Squares_Function, self.Testing_Data['Target'])
		return Residuals, Total_Squares

	def Calculate_Observed_Accuracy(self):
		# This is at at 50% confidence level and picking the item with the highest probability if more than one had a probablility > 50%
		Total_Instances = self.Testing_Data['Target'].count()
		Total_Correct = self.Testing_Data[self.Testing_Data['Target'] == self.Testing_Data['Modified_Results']]['Target'].count()
		self.Observed_Accuracy = (Total_Correct/Total_Instances)

	def Calculate_Kappa_Statistic(self):
		# Calculate_Observed_Accuracy needs to have run prior to this being executed
		Total_Instances = self.Testing_Data['Target'].count()
		Column_Sums = self.Confusion_DF.sum().values
		Row_Sums = self.Confusion_DF.sum(axis=1)
		self.Expected_Accuracy = sum(np.multiply(Column_Sums, Row_Sums)/Total_Instances)/Total_Instances
		self.Kappa_Statistic = (self.Observed_Accuracy - self.Expected_Accuracy)/(1 - self.Expected_Accuracy)

	def Build_DataFrame(self):
		Row_Values = ['AUC','Correlation Coeffecient','R Squared','Relative Absolute Error','Mean Absolute Error','Root Mean Squared Error','Root Relative Squared Error']
		Column_Values = self.Main_Menu.Possible_Classes
		Data = []
		ROC_AUC = copy.deepcopy(self.ROC_AUC)
		ROC_AUC.insert(0,'AUC')
		Data.append(ROC_AUC)
		Correlation_Coefficient = copy.deepcopy(self.Correlation_Coefficient)
		Correlation_Coefficient.insert(0,'Correlation Coefficient')
		Data.append(Correlation_Coefficient)
		R_Squared = copy.deepcopy(self.R_Squared)
		R_Squared.insert(0,'R Squared')
		Data.append(R_Squared)
		Relative_Absolute_Error = copy.deepcopy(self.Relative_Absolute_Error)
		Relative_Absolute_Error.insert(0,'Relative Absolute Error')
		Data.append(Relative_Absolute_Error)
		Mean_Absolute_Error = copy.deepcopy(self.Mean_Absolute_Error)
		Mean_Absolute_Error.insert(0,'Mean Absolute Error')
		Data.append(Mean_Absolute_Error)
		Root_Mean_Squared_Error = copy.deepcopy(self.Root_Mean_Squared_Error)
		Root_Mean_Squared_Error.insert(0,'Root Mean Squared Error')
		Data.append(Root_Mean_Squared_Error)
		Root_Relative_Squared_Error = copy.deepcopy(self.Root_Relative_Squared_Error)
		Root_Relative_Squared_Error.insert(0,'Root Relative Squared Error')
		Data.append(Root_Relative_Squared_Error)
		self.Results_DF = pd.DataFrame(Data)




