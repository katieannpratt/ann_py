import json


class Read_Metadata():

	Metadata_Path = r'./input/metadata/metadata.json'

	def __init__(self):
		with open(self.Metadata_Path,'r') as File:
			JSON_File = json.load(File)
		self.Get_Messages(JSON_File)
		self.Get_Frames(JSON_File)
		self.Get_Activation_Funcations(JSON_File)
		self.Get_Evaluation_Methods(JSON_File)

	def Get_Messages(self, Metadata):
		self.Messages = Metadata["Messages"]

	def Get_Frames(self, Metadata):
		self.Frames = Metadata["Frames"]

	def Get_Activation_Funcations(self, Metadata):
		self.Activation_Functions = Metadata["Activation Functions"]

	def Get_Evaluation_Methods(self, Metadata):
		self.Evaluation_Methods = Metadata["Evaluation Methods"]