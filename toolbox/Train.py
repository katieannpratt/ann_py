import pandas as pd
import random
import numpy as np
import copy

from toolbox import Data_Manipulation

class Backprop_Model(object):

	def __init__(self, parent, Training_Data):
		self.Main_Menu = parent
		self.Training_Data = Training_Data
		self.Prep_Data()
		exec "self.Activation_Function = " + self.Main_Menu.Activation_Function
		exec "self.Activation_Derivative = " + self.Main_Menu.Activation_Derivative
		self.Train()

	def Build_Initial_Weights(self):
		# Initial Weights are created for the Hidden Layer and the Output Layer
		Att_to_Hidden_Weights = []
		Att_to_Hidden_Changes = []
		# Changes are initialized here because they will need to be bassed into the Back_Propgation function in order to apply momentum
		Hidden_to_Output_Weights = []
		Hidden_to_Output_Changes = []
		# Maximum and Minimum weights have been created for each activation function within the metadata to avoid extreme inital random weights
		Minimum = self.Main_Menu.Parent.Metadata.Activation_Functions[self.Main_Menu.Activation_Method]['Weights Range']['Minimum']
		Maximum = self.Main_Menu.Parent.Metadata.Activation_Functions[self.Main_Menu.Activation_Method]['Weights Range']['Maximum']
		Node = 1
		# Weights are then created for each hidden node (this is the node parameter that is selected by the user)
		while Node <= self.Main_Menu.Nodes:
			Hidden_Node_Weights = []
			Hidden_Changes = []
			# Attribute is set to 0 instead of 1 so that an additional Weight is created for Bias
			Attribute = 0
			while  Attribute <= self.Main_Menu.Attribute_Count:
				# Random Weight is generated for each Attribute that is found in the loaded file (Attribute_Count)
				Hidden_Node_Weights.append(random.uniform(Minimum, Maximum))
				Hidden_Changes.append(0)
				Attribute += 1
			Att_to_Hidden_Weights.append(Hidden_Node_Weights)
			Att_to_Hidden_Changes.append(Hidden_Changes)
			Node += 1
		for Output in self.Main_Menu.Possible_Classes:
			Output_Node_Weights = []
			Output_Node_Changes = []
			# Node is set to 0 instead of 1 so that an additional Weight is created for Bias
			Node = 0
			while Node <= self.Main_Menu.Nodes:
				Output_Node_Weights.append(random.uniform(Minimum, Maximum))
				Output_Node_Changes.append(0)
				Node += 1
			Hidden_to_Output_Weights.append(Output_Node_Weights)
			Hidden_to_Output_Changes.append(Output_Node_Changes)
		Weights = [Att_to_Hidden_Weights, Hidden_to_Output_Weights]
		Changes = [Att_to_Hidden_Changes, Hidden_to_Output_Changes]
		# Weights should end up looking like this:
		# [
			# [
				# [H1w1,H1w2,H1b1]
				# ,[H2w1,H2w2,H2b1]
			# ]
			# ,[
				# [O1w1,O1w2,O1b1]
				# ,[O2w1,O2w2, O2b1]
			# ]
		# ]
		return Weights, Changes

	def Prep_Data(self):
		# This function is used to format the Input Data so that the Attributes will be passed in as a vector item, paired with their Target vector item
		self.Epoch = []
		for Index in self.Training_Data.index:
			Inputs = []
			Attribute = 0
			while Attribute < self.Main_Menu.Attribute_Count:
				Inputs.append(self.Training_Data[self.Training_Data.columns[Attribute]][Index])
				Attribute += 1
			Target = self.Training_Data[self.Training_Data.columns[Attribute]][Index]
			self.Epoch.append([Inputs, Target])
				
	def Train(self):
		Iterations = []
		Iteration = 1
		# A hardcoded value of 3 is used here to account for the fluctation that is seen by not defining a seed for my random weight initalizer
		# the weights resulting in the lower Average Sum of Squared Error will be passed back
		while Iteration <= 1:
			# Weights are initalized
			Weights, Changes = self.Build_Initial_Weights()
			E = 1
			while E <= self.Main_Menu.Epochs:
				# Residuals are stored to evaluate comparative performance of each weight set
				Residuals = []
				Targets_T = []
				for Instance in self.Epoch:
					# Outputs are generated using the Feed_Forward process
					Outputs = Data_Manipulation.Feed_Forward(Instance[0], Weights, self.Activation_Function)
					# Weights and Changes are generated using the Back_Propogate process
					Weights, Changes = self.Back_Propogate(Instance[1], Outputs, Weights, Changes)
					Residuals.append(np.subtract(Outputs[-1],Instance[1]))
					Targets_T.append(Instance[1][0])
				RSS = np.sqrt(sum([R * R for R in Residuals]))
				try:
					RSS = sum(RSS)/float(len(RSS))
				except:
					pass
				E += 1
			Iterations.append([RSS, Weights])
			Iteration += 1
		# The best weights are selected based on the RSS value
		self.Weights = sorted(Iterations)[0][1]

	def Back_Propogate(self, Target, Outputs, Weights, Changes):
		# An array for output error is initalized
		Output_Error = []
		# The first loop is for the Output Layer only, iterating through each Output Node
		for Target, Output in zip(Target, Outputs[-1]):
			# An Error is then created for each Output Node by mulitplying the Derivative of the Activation Function at the Output value by the difference between the Target value and the Output value
			Output_Error.append(self.Activation_Derivative(Output)*(Target - Output))
		# Modify Output Weights accordingly
		for i, k in enumerate(zip(Weights[-1], Changes[-1])):
			# Each weight is then looped through to apply the modifications to the weights
			# Applying Weights to the Neuron attribute in the zipped tuple
			Neuron = k[0]
			# Applying Changes from the Changes attribute in the zipped tuple
			Change = k[1]
			for j, Previous_Output in enumerate(Outputs[-2]):
				# Iterating thorugh the Outputs from the Hidden Layers that were passed into the Neuron and applying the weight Updates to the aligning weight
				# The Output Error for the Neuron that was calculated and stored in Output_Error is multiplied by each Output
				# This is then multiplied again by the Learning Rate to modify the rate of learning, then added to the weight to update it
				# Then, the previous Wegiht change of this prior weight is multiplied by the Momentum and this is also added to the weight
				Neuron[j] += (Output_Error[i] * Previous_Output * self.Main_Menu.Learning_Rate) + (Change[j] * self.Main_Menu.Momentum)
				# After the weight has been updated, the Change is stored for the next iteration
				Change[j] = (Output_Error[i] * Previous_Output * self.Main_Menu.Learning_Rate) + (Change[j] * self.Main_Menu.Momentum)
		#Calculate Hidden Node Deltas
		Hidden_Error = []
		for i, Neuron_Output in enumerate(Outputs[-2][:-1]):
			N_Weights = []
			for Output_Neurons in Weights[-1]:
				for j, Assoc_Weight in enumerate(Output_Neurons):
					if j == i:
						N_Weights.append(Assoc_Weight)
			# The error here is a little bit different than the output since we do not have a true target value
			# The Errors from each Output, multiplied by their aligning Weight for a given Hidden Node are summed together
			# And are multiplied by the Derivative of the Activation Function at the point of the Nuerons Output
			# This is the Error of the Hidden Neuron
			Hidden_Error.append(self.Activation_Derivative(Neuron_Output)*(np.dot(Output_Error, N_Weights)))
		# Modify Hidden Layer Weights accordingly
		# Modifications are made to the hidden weights in the same fashion as they were applied to the output weights
		for i, k in enumerate(zip(Weights[-2],Changes[-2])):
			Neuron = k[0]
			Change = k[1]
			for j, Previous_Output in enumerate(Outputs[-3]):
				Neuron[j] += (Hidden_Error[i] * Previous_Output * self.Main_Menu.Learning_Rate) + (Change[j] * self.Main_Menu.Momentum)
				Change[j] = (Hidden_Error[i] * Previous_Output * self.Main_Menu.Learning_Rate) + (Change[j] * self.Main_Menu.Momentum)
		return Weights, Changes

