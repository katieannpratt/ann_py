import tkinter
from tkinter import filedialog as tkFileDialog

from toolbox import GUI_Manipulation
from toolbox import Metadata_Reader
# from toolbox import Data_Manipulation
from toolbox import Frames


class MainApplication(tkinter.Frame):

	def __init__(self, parent):
		tkinter.Frame.__init__(self, parent)
		self.Parent = parent
		self.Metadata = Metadata_Reader.Read_Metadata()
		self.Pack()
		self.Load_Frames()
		self.Metadata.Frames['Main_Menu'].tkraise()

	def Pack(self):
		self.Parent.geometry('{}x{}+{}+{}'.format(800,600,0,0))
		self.Parent.title('Back Propogation Neural Network Application')
		self.Parent.protocol('WM_DELETE_WINDOW',lambda: GUI_Manipulation.Quit_Program(self.Parent))
		
	def Load_Frames(self):
		for F in self.Metadata.Frames:
			exec("self.Metadata.Frames[F] = Frames."+F+"(self)")
			self.Metadata.Frames[F].grid(row=0, column=0, sticky='nsew')


if __name__ == "__main__":
	root = tkinter.Tk()
	MainApplication(root).pack()
	root.mainloop()

raw_input()
