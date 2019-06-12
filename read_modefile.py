import pandas as pd

def read_modefile( kic , header):
	# Python function which loads the modefile of particular RG (define by string <kic>) directly from the 
	# Github repository and returns a Pandas Dataframe containing either the header parameters (header=True) 
	# or the mode parameters (header=None). If the file does not exist None is returned.
	#	Usage:
	#	from read_modefile import read_modefile
	#	freq = read_modefile('1433803', header=None)	#calling function read_modefile to return mode parameters
	#	head = read_modefile('1433803', header=True)	#calling function read_modefile to return header parameters

	url = 'https://raw.githubusercontent.com/tkallinger/KeplerRGpeakbagging/master/ModeFiles/'+kic+'.modes.dat'
	try :
		if header :
			data = pd.read_csv(url, delimiter=' ', skiprows = 5, nrows=1, skipinitialspace = True, header = None, 
				names = ['fmax','fmax_e','dnu','dnu_e','dnu_cor','dnu_cor_e','dnu02','dnu02_e','f0_c','f0_c_e','alpha','alpha_e','evo'] )
		else :
			data = pd.read_csv(url, delimiter=' ', skiprows = 8, skipinitialspace = True, header = None, 
				names = ['degr', 'freq', 'freq_e', 'amp', 'amp_e', 'tau', 'tau_e', 'ev', 'ev1'] )
	except :
		data = None
	return data
