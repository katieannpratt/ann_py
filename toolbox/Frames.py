from __future__ import division
import numpy as np
import Tkinter
import ttk
import tkFileDialog
import tkFont
import random
import copy
import pandas as pd

from toolbox import GUI_Manipulation
from toolbox import Metadata_Reader
from toolbox import Data_Manipulation
from toolbox import Train
from toolbox import Test


class Main_Menu(Tkinter.Frame):

	Evaluation_Method = None
	Activation_Function = None
	Activation_Derivative = None
	Normalized_Neg_1_to_1 = False

	def __init__(self, parent):
		Tkinter.Frame.__init__(self, parent)
		self.Parent = parent
		self.Buttons = []
		self.Pack()
		GUI_Manipulation.Disable_Buttons(self, self.Buttons)
		self.Buttons.append(self.Load_File_Button)

	def Pack(self):
		Body = Tkinter.Frame(self)
		Body.pack(pady=20)
		Top = Tkinter.Frame(self)
		Middle = Tkinter.Frame(self)
		Bottom = Tkinter.Frame(self)

		self.Pack_Top(Top)
		self.Pack_Bottom(Bottom)
		self.Pack_Middle(Middle)
		
	def Pack_Top(self, Location):
		Location.pack(side = 'top')
		Load_File_Label = Tkinter.Label(self, wraplength = 350, text='Please load a text file containing all inputs and the target value.\nThe file should be pipe delimited, and have headers.\nThe last column should contain the target.')
		Load_File_Label.pack(in_ = Location)
		self.Load_File_Button = Tkinter.Button(self, text='Load', width = 20, height = 2, command = self.Load_File)
		self.Load_File_Button.pack(in_ = Location)

	def Pack_Middle(self, Location):
		Location.pack(side='top', pady = 20)
		Middle_Left = Tkinter.Frame(self)
		Middle_Right = Tkinter.Frame(self)
		Middle_Left.pack(side='left', in_ = Location, padx = 40)
		Middle_Right.pack(side='right', in_ = Location, padx = 40)
		self.Pack_Middle_Left(Middle_Left)
		self.Pack_Middle_Right(Middle_Right)

	def Pack_Middle_Left(self, Location):
		AF_Label = Tkinter.Label(self, wraplength = 150, justify = 'left', text = 'Please select your activation function:')
		AF_Label.pack(in_=Location, anchor = 'w')
		Activation_Function_LB = Tkinter.Listbox(self, height = 4, exportselection = False)
		Activation_Function_LB.pack(in_ = Location)
		Activation_Function_LB.bind('<<ListboxSelect>>',self.On_Select)
		for Function in self.Parent.Metadata.Activation_Functions:
			Activation_Function_LB.insert(0, Function)
		EM_Label = Tkinter.Label(self, wraplength = 150, justify = 'left', text = 'Please select your evaluation method:')
		EM_Label.pack(in_ = Location, anchor = 'w')
		Evaluation_Method_LB = Tkinter.Listbox(self, height = 4, exportselection = False)
		Evaluation_Method_LB.pack(in_ = Location)
		Evaluation_Method_LB.bind('<<ListboxSelect>>',self.On_Select)
		for EM in self.Parent.Metadata.Evaluation_Methods:
			Evaluation_Method_LB.insert(0, EM)
		self.Buttons.extend([Activation_Function_LB, Evaluation_Method_LB])

	def Pack_Middle_Right(self, Location):
		First_Section = Tkinter.Frame(self, width = 400)
		Second_Section = Tkinter.Frame(self, width = 400)
		Third_Section = Tkinter.Frame(self, width = 400)
		Fourth_Section = Tkinter.Frame(self, width = 400)
		Fifth_Section = Tkinter.Frame(self, width = 400)
		Sixth_Section = Tkinter.Frame(self, width = 400)

		First_Section.pack(in_ = Location, side='top', fill='both', pady = 5)
		Second_Section.pack(in_ = Location, side='top', fill='both', pady = 5)
		Third_Section.pack(in_ = Location, side='top', fill='both', pady = 5)
		Fourth_Section.pack(in_=Location, side='top', fill='both', pady = 5)
		Fifth_Section.pack(in_=Location, side='top', fill='both', pady = 5)
		Sixth_Section.pack(in_=Location, side='top', fill='both', pady = 5)

		HN_Label = Tkinter.Label(self, wraplength = 200, justify = 'left', text = 'Number of hidden nodes:')
		HN_Label.pack(in_ = First_Section, side = 'left', anchor = 'w')
		self.Hidden_Nodes_Entry = Tkinter.Entry(self, width = 5)
		self.Hidden_Nodes_Entry.pack(in_ = First_Section, side = 'right', padx = 5, anchor = 'e')

		EP_Label = Tkinter.Label(self, wraplength = 200, justify = 'left', text = 'Number of epochs to train with:')
		EP_Label.pack(in_ = Second_Section, side = 'left', anchor = 'w', fill = 'both')
		self.Epochs_Entry = Tkinter.Entry(self, width = 5)
		self.Epochs_Entry.pack(in_ = Second_Section, side = 'right', padx = 5, anchor = 'e')

		LR_Label = Tkinter.Label(self, wraplength = 200, justify = 'left', text = 'Learning Rate:')
		LR_Label.pack(in_ = Third_Section, side = 'left', anchor = 'w', fill = 'both')
		self.Learning_Rate_Entry = Tkinter.Entry(self, width = 5)
		self.Learning_Rate_Entry.pack(in_ = Third_Section, side = 'right', padx = 5, anchor = 'e')
		self.Learning_Rate_Entry.insert(0,'1')

		M_Label = Tkinter.Label(self, wraplength = 200, justify = 'left', text = 'Momentum:')
		M_Label.pack(in_ = Fourth_Section, side = 'left', anchor = 'w', fill = 'both')
		self.Momentum_Entry = Tkinter.Entry(self, width = 5)
		self.Momentum_Entry.pack(in_ = Fourth_Section, side = 'right', padx = 5, anchor = 'e')
		self.Momentum_Entry.insert(0,'0')

		Percent_Label = Tkinter.Label(self, wraplength = 200, justify = 'left', text = 'Percent of data used for training:')
		Percent_Label.pack(in_ = Fifth_Section, side = 'left', anchor = 'w', fill = 'both')
		self.Percent_Entry = Tkinter.Entry(self, width = 5)
		self.Percent_Entry.pack(in_ = Fifth_Section, side = 'right', padx = 5, anchor = 'e')

		Folds_Label = Tkinter.Label(self, wraplength = 200, justify = 'left', text = 'Number of folds used in cross-validation:')
		Folds_Label.pack(in_ = Sixth_Section, side = 'left', anchor = 'w', fill = 'both')
		self.Folds_Entry = Tkinter.Entry(self, width = 5)
		self.Folds_Entry.pack(in_ = Sixth_Section, side = 'right', padx = 5, anchor = 'e')

		self.Buttons.extend([self.Hidden_Nodes_Entry, self.Epochs_Entry, self.Percent_Entry, self.Folds_Entry, self.Learning_Rate_Entry, self.Momentum_Entry])
		
	def Pack_Bottom(self, Location):
		Location.pack(side='bottom', pady = 20)
		Run_Button = Tkinter.Button(self, text = 'Run', width = 20, height = 2, command = self.Run_Neural_Network)
		Run_Button.pack(in_ = Location)
		self.Buttons.append(Run_Button)

	def On_Select(self, evt):
		w = evt.widget
		index = int(w.curselection()[0])
		value = w.get(index)
		if value in self.Parent.Metadata.Evaluation_Methods:
			for Statement in self.Parent.Metadata.Evaluation_Methods[value]['Response']:
				exec Statement
			self.Evaluation_Method = value
		elif value in self.Parent.Metadata.Activation_Functions:
			for Statement in self.Parent.Metadata.Activation_Functions[value]['Response']:
				exec Statement
			self.Activation_Function = self.Parent.Metadata.Activation_Functions[value]['Function']
			self.Activation_Derivative = self.Parent.Metadata.Activation_Functions[value]['Derivative']
			self.Activation_Method = value

	def Load_File(self):
		File_Path = tkFileDialog.askopenfilename(title='Select Pipe Delimited to Load')
		self.DataFrame, self.Attribute_Count, self.Instance_Count, self.Normalization, self.Possible_Classes, self.DataSet_Builder = Data_Manipulation.Load_New_Text_File(File_Path)
		GUI_Manipulation.Enable_Buttons(self, self.Buttons)
		GUI_Manipulation.Disable_Buttons(self, [self.Percent_Entry, self.Folds_Entry])

	def Run_Neural_Network(self):
		if self.Evaluation_Method and self.Activation_Function and self.Hidden_Nodes_Entry.get() and self.Epochs_Entry.get() and self.Momentum_Entry.get() and self.Learning_Rate_Entry.get():
			try:
				if np.int(self.Hidden_Nodes_Entry.get()) > 0 and np.int(self.Epochs_Entry.get()) > 0 and (0 < np.float(self.Learning_Rate_Entry.get()) <= 5) and (0 <= np.float(self.Momentum_Entry.get()) <= 1):
					self.Nodes = np.int(self.Hidden_Nodes_Entry.get())
					self.Epochs = np.int(self.Epochs_Entry.get())
					self.Learning_Rate = np.float(self.Learning_Rate_Entry.get())
					self.Momentum = np.float(self.Momentum_Entry.get())
				else:
					GUI_Manipulation.Message(self.Parent.Metadata.Messages['Invalid Value'])
			except:
				GUI_Manipulation.Message(self.Parent.Metadata.Messages['Invalid Value'])
				return
			if self.Parent.Metadata.Evaluation_Methods[self.Evaluation_Method]['Required']:
				for Required in self.Parent.Metadata.Evaluation_Methods[self.Evaluation_Method]['Required']:
					exec Required.keys()[0]
					if Result:
						pass
					else:
						Message = Required[Required.keys()[0]]
						GUI_Manipulation.Message(self.Parent.Metadata.Messages[Message])
						return
				self.Call_BackPropagration()
		else:
			GUI_Manipulation.Message(self.Parent.Metadata.Messages['Missing Information'])

	def Call_BackPropagration(self):
		if self.Evaluation_Method == 'Random Split':
			self.Percent = np.int(self.Percent_Entry.get())
			Sample_Size = np.int(self.Instance_Count * (self.Percent/100))
			Rows = random.sample(self.DataFrame.index, Sample_Size)
			Training_Data = self.DataFrame.ix[Rows]
			Testing_Data = self.DataFrame.drop(Rows)
			Model = Train.Backprop_Model(self, Training_Data)
			self.Results = Test.Test_Data(self, Testing_Data, Model.Weights)
			self.Results.Evaluate(self.Results.Testing_Data)
		elif self.Evaluation_Method == 'Training Data':
			Model = Train.Backprop_Model(self, self.DataFrame)
			self.Results = Test.Test_Data(self, self.DataFrame, Model.Weights)
			self.Results.Evaluate(self.Results.Testing_Data)
		elif self.Evaluation_Method == 'Cross-validation':
			self.Folds = np.int(self.Folds_Entry.get())
			Testing_Rows, Training_Rows = Data_Manipulation.Cross_Validation(self.DataFrame.index, self.Folds)
			Aggregating_Data = pd.DataFrame()
			for Train_Rows, Test_Rows in zip(Training_Rows, Testing_Rows):
				Training_Data = self.DataFrame.ix[Train_Rows]
				Testing_Data = self.DataFrame.ix[Test_Rows]
				Model = Train.Backprop_Model(self, Training_Data)
				self.Results = Test.Test_Data(self, Testing_Data, Model.Weights)
				Aggregating_Data = Aggregating_Data.append(self.Results.Testing_Data)
			self.Results.Evaluate(Aggregating_Data)
		elif self.Evaluation_Method == 'Stratified Split':
			self.Percent = np.int(self.Percent_Entry.get())
			Rows = []
			for Class in self.Possible_Classes:
				Sample_Size = np.int(self.DataSet_Builder[(self.DataSet_Builder['Target']==Class)]['Target'].count() * (self.Percent/100))
				Rows.extend(random.sample(self.DataSet_Builder[self.DataSet_Builder['Targetl']==Class].index, Sample_Size))
			Training_Data = self.DataFrame.ix[Rows]
			Testing_Data = self.DataFrame.drop(Rows)
			Model = Train.Backprop_Model(self, Training_Data)
			self.Results = Test.Test_Data(self, Testing_Data, Model.Weights)
			self.Results.Evaluate(self.Results.Testing_Data)
		self.Parent.Metadata.Frames['Evaluation_View'].Pull_Values()
		self.Parent.Metadata.Frames['Evaluation_View'].tkraise()
		# # Remove This After use
		# self.Results.Testing_Data.to_csv('results.txt',sep='|')

class Evaluation_View(Tkinter.Frame):

	def __init__(self, parent):
		Tkinter.Frame.__init__(self, parent)
		self.Parent = parent
		self.Buttons = []
		self.Pack()

	def Pack(self):
		Body = Tkinter.Frame(self)
		Body.pack(pady=20)

		Top = Tkinter.Frame(self)
		self.Middle = Tkinter.Frame(self)
		Bottom = Tkinter.Frame(self)

		Top.pack(in_ = Body)
		self.Middle.pack(in_ = Body)
		Bottom.pack(in_ = Body)

		self.Accuracy_Label = Tkinter.Label(self, text = 'Observed Accuracy: None')
		self.Accuracy_Label.pack(in_ = Top)

		self.Kappa_Label = Tkinter.Label(self, text = 'Kappa Statistic: None')
		self.Kappa_Label.pack(in_ = Top)

		self.R2_Label = Tkinter.Label(self, text = 'R Squared: None')
		self.R2_Label.pack(in_ = Top)

		self.Rel_Abs_Error_Label = Tkinter.Label(self, text = 'Relative Absolute Error: None')
		self.Rel_Abs_Error_Label.pack(in_=Top)

		self.Mean_Abs_Error_Label = Tkinter.Label(self, text = 'Mean Absolute Error: None')
		self.Mean_Abs_Error_Label.pack(in_ = Top)

		self.Root_Mean_Sq_Error_Label = Tkinter.Label(self, text = 'Root Mean Squared Error: None')
		self.Root_Mean_Sq_Error_Label.pack(in_=Top)

		self.Root_Rel_Sq_Error_Label = Tkinter.Label(self, text = 'Root Relative Squared Error: None')
		self.Root_Rel_Sq_Error_Label.pack(in_=Top)

		Confusion_Matrix_Button =  Tkinter.Button(self, width = 30, height = 2, text = 'View Normalized Confusion Matrix', command = self.Call_Confusion_Matrix)
		Confusion_Matrix_Button.pack(in_ = Bottom)

		ROC_Curves_Button = Tkinter.Button(self, width = 30, height = 2, text = 'View ROC Curves', command = self.Call_ROC_Curves)
		ROC_Curves_Button.pack(in_=Bottom)

		Return_Button = Tkinter.Button(self, width = 20, height = 2, text = 'Return to Main Menu', command = self.Return_Main_Menu)
		Return_Button.pack(in_=Bottom)

	def Pull_Values(self):
		GUI_Manipulation.Update_Label(self, self.Accuracy_Label, 'Observed Accuracy: %0.4f' % self.Parent.Metadata.Frames['Main_Menu'].Results.Observed_Accuracy)
		GUI_Manipulation.Update_Label(self, self.Kappa_Label, 'Kappa Statistic: %0.4f' % self.Parent.Metadata.Frames['Main_Menu'].Results.Kappa_Statistic)
		GUI_Manipulation.Update_Label(self, self.R2_Label, 'R Squared: %0.4f' % self.Parent.Metadata.Frames['Main_Menu'].Results.Overall_R_Squared)
		GUI_Manipulation.Update_Label(self, self.Rel_Abs_Error_Label, 'Relative Absolute Error: %0.4f' % self.Parent.Metadata.Frames['Main_Menu'].Results.Overall_Relative_Absolute_Error)
		GUI_Manipulation.Update_Label(self, self.Mean_Abs_Error_Label, 'Mean Absolute Error: %0.4f' % self.Parent.Metadata.Frames['Main_Menu'].Results.Overall_Mean_Absolute_Error)
		GUI_Manipulation.Update_Label(self, self.Root_Mean_Sq_Error_Label, 'Root Mean Squared Error: %0.4f' % self.Parent.Metadata.Frames['Main_Menu'].Results.Overall_Root_Mean_Squared_Error)
		GUI_Manipulation.Update_Label(self, self.Root_Rel_Sq_Error_Label, 'Root Relative Squared Error: %0.4f' % self.Parent.Metadata.Frames['Main_Menu'].Results.Overall_Root_Relative_Squared_Error)
		self.Pack_TkTree()

	def Call_Confusion_Matrix(self):
		Data_Manipulation.Plot_Confusion_Matrix(self.Parent.Metadata.Frames['Main_Menu'].Results.Confusion_Norm_DF, 'Normalized Confusion Matrix')

	def Return_Main_Menu(self):
		self.Parent.Metadata.Frames['Main_Menu'].tkraise()

	def Call_ROC_Curves(self):
		Data_Manipulation.Plot_ROC_Curves(self.Parent.Metadata.Frames['Main_Menu'].Results.FPR, self.Parent.Metadata.Frames['Main_Menu'].Results.TPR, self.Parent.Metadata.Frames['Main_Menu'].Results.ROC_AUC, self.Parent.Metadata.Frames['Main_Menu'].Possible_Classes)

	def Pack_TkTree(self):
		DataFrame = self.Parent.Metadata.Frames['Main_Menu'].Results.Results_DF
		Columns = [np.str(x) for x in copy.deepcopy(self.Parent.Metadata.Frames['Main_Menu'].Possible_Classes)]
		Columns.insert(0,'Evaluation Statistics')
		self.Results = ttk.Treeview(self, columns=Columns, show='headings')
		self.Column_Formatting(Columns, self.Results)
		Data_Manipulation.Insert_Records(DataFrame,self.Results)
		self.Results.grid(column=0, row=0, sticky = 'nsew', in_ = self.Middle, pady=10)

	def Column_Formatting(self, Columns, Object):
		for col in Columns:
			Object.heading(col, text=col.title())
			Object.column(col, width=max(tkFont.Font().measure(col.title()),105))