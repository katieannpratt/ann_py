import os
import pandas as pd
import numpy as np
import itertools
import copy
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def Load_New_Text_File(File_Path):
	DataFrame = pd.io.parsers.read_csv(File_Path, sep='|')
	Attribute_Count = len(DataFrame.columns) - 1
	Instance_Count = DataFrame[DataFrame.columns[-1:]].count().values[0]
	i = 0
	Normalization = False
	while i <= (Attribute_Count - 1):
		Max_Value = max(DataFrame[DataFrame.columns[i]])
		Min_Value = min(DataFrame[DataFrame.columns[i]])
		if Max_Value > 1 or Min_Value < 0:
			Normalize_Function = lambda x: np.float((x-Min_Value))/(Max_Value-Min_Value)
			DataFrame[DataFrame.columns[i]] = map(Normalize_Function,DataFrame[DataFrame.columns[i]])
			Normalization = True
		i += 1
	Possible_Classes = []
	DataSet_Builder = DataFrame[[DataFrame.columns[-1]]]
	DataSet_Builder.columns = ['Target']
	if sorted(DataFrame[DataFrame.columns[i]].unique()) == [0,1]:
		Possible_Classes = [1]
	else:
		for Value in DataFrame[DataFrame.columns[i]].unique():
			Possible_Classes.append(Value)
	DataFrame[DataFrame.columns[i]] = [Convert_to_Array(x,Possible_Classes) for x in DataFrame[DataFrame.columns[i]]]
	DataFrame = DataFrame.rename(columns = {DataFrame.columns[-1]:'Target'})
	return DataFrame, Attribute_Count, Instance_Count, Normalization, Possible_Classes, DataSet_Builder

def Convert_to_Array(String, Possible_Classes):
	New_Item = []
	for Value in Possible_Classes:
		if String == Value:
			New_Item.insert(Possible_Classes.index(Value),1)
		else:
			New_Item.insert(Possible_Classes.index(Value),0)
	return New_Item

def Normalize_Neg_1_to_1(DataFrame, Attribute_Count, Normalized_Neg_1_to_1):
	if Normalized_Neg_1_to_1:
		pass
	else:
		i = 0
		while i <= (Attribute_Count - 1):
			Normalize_Function = lambda x: -1 + (2*x)
			DataFrame[DataFrame.columns[i]] = map(Normalize_Function, DataFrame[DataFrame.columns[i]])
			i += 1
		Normalize_Function = lambda x: [-1 + (2*a) for a in x]
		DataFrame[DataFrame.columns[i]] = map(Normalize_Function, DataFrame[DataFrame.columns[i]])
	return DataFrame, True

def Normalize_0_to_1(DataFrame, Attribute_Count, Normalized_Neg_1_to_1):
	if Normalized_Neg_1_to_1:
		i = 0
		while i <= (Attribute_Count - 1):
			Normalize_Function = lambda x: (x+1)/2
			DataFrame[DataFrame.columns[i]] = map(Normalize_Function, DataFrame[DataFrame.columns[i]])
			i += 1
		Normalize_Function = lambda x: [(a+1)/2 for a in x]
		DataFrame[DataFrame.columns[i]] = map(Normalize_Function, DataFrame[DataFrame.columns[i]])
	return DataFrame, False

def Feed_Forward(Inputs, Weights, Activation_Function):
	# Initialize Array to Contain the Outputs at each Perceptron
	Outputs = []
	# Inputs is initialized using the original inputs - this is an array of the attributes submitted in the files for a particular instance
	Inputs = copy.deepcopy(Inputs)
	# The first loop is for every layer in the weights, beginning first with the hidden node calculations and then moving to the output layer
	for Layer in Weights:
		# A value of 1 is added to the Input to account for Bias
		Inputs.append(1)
		# The current inputs represent the outputs for the prior layer (with bias)
		Outputs.append(Inputs)
		Layer_Output = []
		# The second loop is an iterations thorugh each node in the layer
		for Node in Layer:
			# The Output of a particular node is the dot product of the Input Vector and Weight Vector that is then inputted into the given activation function
			Output = Activation_Function(np.dot(Inputs,Node))
			# This value is then appended ot the Layer_Output (there should be a single value appended for each node, creating a new vector array to be passed through to the next layer)
			Layer_Output.append(Output)
		# The input layer is then redefined as the Output of the prior layer
		Inputs = Layer_Output
	# Finally, the output layer values are appended to the array, resulting in a final array of a length of 3 (input, hidden, output), each contianing an array of each layer's outputs
	Outputs.append(Layer_Output)
	return Outputs

def Plot_Confusion_Matrix(Confusion_Matrix, Title):
	plt.matshow(Confusion_Matrix, cmap = plt.cm.gray_r, vmin=0, vmax=1)
	plt.colorbar()
	Tick_Marks = np.arange(len(Confusion_Matrix.columns))
	plt.xticks(Tick_Marks, Confusion_Matrix.columns, rotation=45)
	plt.yticks(Tick_Marks, Confusion_Matrix.index)
	plt.ylabel(Confusion_Matrix.index.name)
	plt.xlabel(Confusion_Matrix.columns.name)
	plt.show()

def Plot_ROC_Curves(False_Positive_Rate, True_Positive_Rate, ROC_AUC, Possible_Classes):
	plt.clf()
	plt.plot([0, 1], [0, 1], 'k--', label='Neutral Line')
	for FPR, TPR, AUC, Class in zip(False_Positive_Rate, True_Positive_Rate, ROC_AUC, Possible_Classes):
		plt.plot(FPR, TPR, label=np.str(Class) + ' (area = %0.2f)' % AUC, lw = 1.7)
	plt.xlim([-0.005, 1.0])
	plt.ylim([0.0, 1.005])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver Operating Characteristic')
	plt.legend(loc="lower right")
	plt.show()

def Insert_Records(Data, Object):
	for item in Object.get_children():
		Object.delete(item)
	Function = lambda x: [x[a][Index] for a in x.columns]
	for Index in Data.index:
		Object.insert('','end',values=Function(Data))

def Cross_Validation(Items, Folds):
	Testing_Data = []
	Training_Data = []
	Slices = [Items[i::Folds] for i in xrange(Folds)]
	for i in xrange(Folds):
		Testing_Data.append(Slices[i])
		Training_Data.append([Item for S in Slices if S is not Testing_Data for Item in S])
	return Testing_Data, Training_Data