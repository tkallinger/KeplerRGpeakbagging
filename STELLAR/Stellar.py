__author__ = 'thomas kallinger'
__version__ = '1'

import pandas as pd
import numpy as np 
import math
from scipy.stats import pearsonr
from uncertainties import ufloat
from uncertainties import ufloat_fromstr
import shutil
import os
import json
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots
plt.style.use(['science'])
plt.rcParams.update({'font.size':8})

from ultranest import ReactiveNestedSampler
from ultranest.plot import cornerplot

################################################################################################
################################################################################################
##############              HELPER CLASS : manages UltraNest In/Output       ###################
################################################################################################
################################################################################################
class NestedFitter():
	def __init__(self, x, y, ye, parameters, par_lo, par_hi, function, 
				prior=None, wrapped_params=None, return_logZ=False, 
				MLE=False, verbose=True):
		# Convert inputs to numpy arrays for vectorized operations
		self.x = np.asarray(x)
		self.y = np.asarray(y)
		self.ye = np.asarray(ye)

		self.parameters = parameters
		self.par_lo = np.asarray(par_lo)
		self.par_hi = np.asarray(par_hi)
		self.function = function
		self.prior = prior
		self.MLE = MLE
		self.return_logZ = return_logZ
		self.wrapped_params = wrapped_params if wrapped_params else [False] * len(parameters)
		self.verbose = verbose
        # Precompute values that don't change
		self.par_ranges = self.par_hi - self.par_lo
		self.inv_ye_sq = 1.0 / (self.ye ** 2)  # For MLE calculations

	def prior_transform(self, cube):
		"""Vectorized prior transform"""
		return self.par_lo + cube * self.par_ranges

	def log_likelihood(self, params):
		"""Optimized log likelihood calculation"""
		y_model = self.function(self.x, *params)
		residuals = y_model - self.y
		loglike = -0.5 * np.sum(residuals * residuals * self.inv_ye_sq)
        
		if self.prior is not None:
			loglike += self.prior(*params)
		return loglike

	def log_likelihood_MLE(self, params):
		"""Optimized MLE log likelihood"""
		y_model = self.function(self.x, *params)
		ratio = self.y / y_model
		loglike = -np.sum(ratio + np.log(y_model))
		if self.prior is not None:
			idx = self.prior[0]
			loglike += - 0.5 * (params[idx] - self.prior[1])**2 / self.prior[2]**2
		return loglike

	def fitter(self, plots=False, n_live_points=200):
		"""Optimized fitting routine"""
		if os.path.exists('sampler'):
			shutil.rmtree('sampler')
            
		# Choose the appropriate likelihood function
		loglike = self.log_likelihood_MLE if self.MLE else self.log_likelihood
        
		# Initialize sampler with vectorized functions
		sampler = ReactiveNestedSampler(
			self.parameters, 
			loglike, 
			self.prior_transform,
			log_dir="sampler", 
			resume=True, 
			wrapped_params=self.wrapped_params
		)
        
		# Run with optimized parameters
		viz = 'auto' if self.verbose else False
		run_kwargs = {
			'min_num_live_points': n_live_points,
			'dKL': np.inf,
			'min_ess': 100,
			'show_status': self.verbose,
			'viz_callback': viz
		}
        
		result = sampler.run(**run_kwargs)
        
		if plots:
			sampler.plot()
            
		if self.verbose:
			sampler.print_results()
            
		# Process results
		post = result['posterior']
		par = np.array(post['median'])
		par_up = np.abs(post['errup'] - par)
		par_lo = np.abs(par - post['errlo'])
		par_e = 0.5 * (par_up + par_lo)
        
		return (par, par_e, result['logz']) if self.return_logZ else (par, par_e)



################################################################################################
################################################################################################
##############    MAIN CLASS : STELLAR : Solar-TypE osciLLation AnalyseR     ###################
################################################################################################
################################################################################################
class Stellar():
	def __init__(self, ID, pds, path='files', f_nyg=4165.2, verbose=True):
		#if not isinstance(pds, pd.DataFrame):
		#	print('pds is not a pandas dataframe -> exit')
		#	quit()
		self.ID = ID
		self.path = '{}/{}'.format(path,ID)
		self.pds = pds						#pandas obj with columns 'f' and 'p'
		self.f_nyg = f_nyg
		self.verbose = verbose
		self.loggerfile = self.path+'/'+self.ID+'.log'
		if not os.path.exists(self.path) : os.mkdir(self.path)
		if os.path.isfile(self.loggerfile): os.remove(self.loggerfile)

	def logger(self,message):
		with open(self.loggerfile,'a') as f:
			f.write(message+'\n')

	def eta(self, f):
		arg = (np.pi / 2 * f / self.f_nyg)
		return (np.sin(arg)/arg)**2

	def crosscorr(self, x, y, y1, frange):
		df = x[1] - x[0]
		f_a = np.arange(frange[0], frange[1], df)
		p_a = np.zeros(len(f_a))
		ind = np.arange(len(f_a)) + round(frange[0]/df)
		for i in range(len(f_a)):
			p_a[i] = np.corrcoef(y, np.roll(y1,int(ind[i])))[0, 1]
		return f_a, p_a**2

	def lp(self, x, f, lifetime, H_c, f_c, w_c):
		#lw = 11.57407 / (np.pi*lifetime)
		#H_c = a_c**2 / (np.pi * lw)
		#return H_c * np.exp((f - f_c)**2 / (-2*w_c**2)) / (1 + 4*((x - f)/lw)**2)

		inv_lifetime = 11.57407 / (np.pi * lifetime)  # 11.57407 = 1/(86400*1e-6)
		lw = inv_lifetime
		inv_lw = 1.0 / lw
    
		f_diff = f - f_c
		gaussian = np.exp(-0.5 * (f_diff * f_diff) / (w_c * w_c))
    
		x_diff = x - f
		lorentzian = 1.0 / (1.0 + 4.0 * (x_diff * inv_lw) ** 2)
    
		return H_c * gaussian * lorentzian
	
	def lor02(self, x, f0, a0, tau0, f2, a2, tau2, use_bg=True):
		lw0 = 11.57407 / (np.pi*tau0)
		lw2 = 11.57407 / (np.pi*tau2)
		model =  a0**2 / (lw0*np.pi) / (1 + 4*((x-f0)/lw0)**2) + a2**2 / (lw2*np.pi) / (1 + 4*((x-f2)/lw2)**2)
		if use_bg: model += self.pds_now.bgfit
		return model 

	def lor02_rot(self, x, f0, a0, tau0, f2, a2, tau2, frot, incl, use_bg=True):
		lw0 = 11.57407 / (np.pi*tau0)
		lw2 = 11.57407 / (np.pi*tau2)
		rad = np.radians(incl)
		h = a2**2 / (lw2*np.pi)
		hc = h * 0.25 * (3*np.cos(rad)**2 - 1)**2
		hs_1 = h * 3/8 * np.sin(2*rad)**2
		hs_2 = h * 3/8 * np.sin(rad)**4
		model =  a0**2 / (lw0*np.pi) / (1 + 4*((x-f0)/lw0)**2) + \
			hc / (1 + 4*((x-f2)/lw2)**2) + \
			hs_1 / (1 + 4*((x-f2-frot)/lw2)**2) + \
			hs_1 / (1 + 4*((x-f2+frot)/lw2)**2) + \
			hs_2 / (1 + 4*((x-f2-2*frot)/lw2)**2) + \
			hs_2 / (1 + 4*((x-f2+2*frot)/lw2)**2) 
		if use_bg: model += self.pds_now.bgfit
		return model 

	def lor(self, x, f, a, tau, use_bg=True):
		lw = 11.57407 / (np.pi*tau)
		model =  a**2 / (lw*np.pi) / (1 + 4*((x-f)/lw)**2)
		if use_bg: model += self.pds_now.bgfit
		return model 

	def lor_rot(self, x, f, a, tau, frot, incl, use_bg=True):
		lw = 11.57407 / (np.pi*tau)
		h = a**2 / (lw*np.pi)
		rad = np.radians(incl)
		hc = h * (np.cos(rad))**2
		hs = h * 0.5 * (np.sin(rad))**2
		model = hc / (1 + 4*((x-f)/lw)**2) + \
				hs / (1 + 4*((x-f-frot)/lw)**2) + \
				hs / (1 + 4*((x-f+frot)/lw)**2) 
		if use_bg: model += self.pds_now.bgfit
		return model 

	def sinc(self, x, f, a, use_bg=True):
		df = x[1] - x[0]
		arg = np.pi * (x - f) / df
		model = 2 * a**2 / df * (np.sin(arg) / arg)**2
		if use_bg: model += self.pds_now.bgfit
		return model

	def LorModel(self, f, fm, dnu, dnu02, dnu01, amp0, amp1, amp2):
		dnu02 *= dnu
		dnu01 *= dnu
		tau = self.tau_guess.n
		f_env = self.bg_parameter['fmax'].n
		sig_env = self.bg_parameter['sig'].n
		model = self.pds.bgfit + \
			self.lp(f, fm, tau, amp0, f_env, sig_env) + \
			self.lp(f, fm - dnu, tau, amp0, f_env, sig_env) + \
			self.lp(f, fm + dnu, tau, amp0, f_env, sig_env) + \
			self.lp(f, fm - dnu02, tau, amp2, f_env, sig_env) + \
			self.lp(f, fm - dnu - dnu02, tau, amp2, f_env, sig_env) + \
			self.lp(f, fm + dnu - dnu02, tau, amp2, f_env, sig_env) + \
			self.lp(f, fm - dnu/2 + dnu01, tau, amp1, f_env, sig_env) + \
			self.lp(f, fm + dnu/2 + dnu01, tau, amp1, f_env, sig_env)  
		return model

	def BGmodel(self, f, n, Pg, fmax, sig, a1, b1, a2, b2):
		zeta = 2 * np.sqrt(2) / np.pi
		model = n + self.eta(f) * (Pg * np.exp(-(f-fmax)**2 / (2 * sig**2)) +
			zeta * a1**2 / b1 / (1 + (f / b1)**4) +
			zeta * a2**2 / b2 / (1 + (f / b2)**4) )
		return model

	def para_read(self, file):
		with open(file) as f:
			d = f.read()
		s= d.replace('{','').replace('}','').replace("'","").replace(' ','').split(',')
		struc = {}
		for d in s:
			d = d.split(':')
			key = d[0]
			value = d[1] if key in ['probability','RGB-prob','RC-prob','2ndRC-prob','AGB-prob'] else ufloat_fromstr(d[1])
			struc[key] = value
		return struc

	def init_function(self, pds=True, bg_par=False, dnu_par=False, freq=False):
		if pds: self.pds = pd.read_csv(self.path+'/'+self.ID+'.bg.fit.pds')
		if bg_par: self.bg_parameter =  self.para_read(self.path+'/'+self.ID+'.bg_par.dat')
		if dnu_par: 
			self.dnu_parameter = self.para_read(self.path+'/'+self.ID+'.dnu_par.dat')
			for key, value in self.dnu_parameter.items():
				if 'prob' in key:
					self.evo = 'RGB' if 'RGB' in key else 'RC'

		if freq: 
			filename = self.path+'/'+self.ID+'.modes.dat'
			self.freq = pd.read_csv(filename, sep=' ', skiprows=6, skipinitialspace=True)
			with open(filename, 'r') as fn:
				self.freq_header = fn.readlines()[0:6] 

	def evi2prob(self, lgZ):
		lgZ -= np.min(lgZ)
		while np.max(lgZ) > 500:
			lgZ /= 1.2 
			#print(lgZ)
		p = []
		for i in range(len(lgZ)):
			p.append(np.exp(lgZ[i])/np.exp(lgZ).sum())
		return p

	def fmax_estimator(self, plot=False):
		pds = self.pds.copy()
		if plot: plt.plot(pds.f,pds.p)
		frange = 10**np.linspace(0,np.log10(pds.f.max()),100)
		frange = np.insert(frange,0,0,axis=0)
		#f,p = [],[]
		for i in range(len(frange)-1):
			idx = pds.index[(pds.f > frange[i]) & (pds.f < frange[i+1])].tolist()
			#f.append(np.mean(pds.loc[idx,'f']))
			#p.append(np.median(pds.loc[idx,'p']))
			pds.loc[idx,'p'] /= np.median(pds.loc[idx,'p'])
		ff = np.linspace(0,int(pds.f.max()),2*int(pds.f.max()))
		cc = np.zeros(len(ff))
		for i in range(1,len(ff)):
			g = np.exp(-(pds.f.values-ff[i])**2/2/(0.15*ff[i])**2) / (0.15*ff[i]*np.sqrt(2*np.pi))
			cc[i] = pearsonr(g,pds.p.values)[0]
		fmax_guess = ff[np.argmax(cc)]
		if plot:
			plt.plot(pds.f,pds.p)
			plt.axvline(fmax_guess,color='red')
			plt.show()
		return fmax_guess

################################################################################################
##################### Fit Granulation Background ###############################################
################################################################################################
	def fmax_fitter(self, fmax_guess=False, plot=False):
		start_time = time.time()
		self.fmax_guess = fmax_guess if fmax_guess else self.fmax_estimator()

		self.pds = self.pds[(self.pds.f > 0.1*self.fmax_guess) & (self.pds.f < 10*self.fmax_guess)].reset_index(drop=True)

		n = self.pds['p'][self.pds.f > 0.9*self.pds.f.max()].mean()
		sig = 20.0*(1.0-np.exp(-self.fmax_guess/100.)) if self.fmax_guess < 300 else self.fmax_guess/10
		amp = 3382.*self.fmax_guess**(-0.609) #+/- 50%
		b1 = 0.317*self.fmax_guess**0.970
		b2 = 0.948*self.fmax_guess**0.992
		Pg = self.pds.p.max()

		parameters=['n', 'Pg', 'fmax', 'sig', 'a1', 'b1', 'a2', 'b2']
		fit = NestedFitter(
			x=self.pds.f.values, 
			y=self.pds.p.values, 
			ye=[], 
			parameters=parameters, 
			par_lo=[0.5*n, 0, 0.8*self.fmax_guess,0.5*sig,0.1*amp,0.5*b1,0.1*amp,0.5*b2], 
			par_hi=[2.0*n,Pg, 1.2*self.fmax_guess,2.0*sig,2.0*amp,2.0*b1,2.0*amp,2.0*b2],
			function=self.BGmodel,
			MLE=True,
			return_logZ=True,
			verbose=self.verbose)

		c,c_e,lg_z = fit.fitter()
		self.pds['fit'] = self.BGmodel(self.pds.f.values, *c)
		cc = np.copy(c)
		cc[0], cc[1], cc[6] = 0, 0, 0
		self.pds['lfit1'] = self.BGmodel(self.pds.f.values, *cc)
		cc = np.copy(c)
		cc[0], cc[1], cc[4] = 0, 0, 0
		self.pds['lfit2'] = self.BGmodel(self.pds.f.values, *cc)
		cc = np.copy(c)
		cc[1] = 0
		self.pds['bgfit'] = self.BGmodel(self.pds.f.values, *cc)
		
		self.pds.to_csv(self.path+'/'+self.ID+'.bg.fit.pds', index=False)

		Ptot = (c[3] + np.random.normal(0,c_e[3],1000)) * (c[1]+ np.random.normal(0,c_e[1],1000) ) * np.sqrt(2*np.pi)
		Ptot = np.sqrt(Ptot[np.argwhere(Ptot > 0)])
		c   = np.append(c,np.mean(Ptot))
		c_e = np.append(c_e,np.std(Ptot))

		lg_z_bgfit = - np.sum(self.pds.p.values/self.pds.bgfit.values + np.log(self.pds.bgfit.values))
		z = [lg_z,lg_z_bgfit]
		z -= np.min(z)

		if np.max(z) < 100:
			propability = np.exp(z[0])/np.sum(np.exp(z))
		else:
			propability = 1

		self.bg_parameter = {}
		parameters.append('Ptot')
		for i in range(len(parameters)):
			self.bg_parameter[parameters[i]] = ufloat(c[i],c_e[i])
		self.bg_parameter['probability'] = propability
		print(self.bg_parameter)
		with open(self.path+'/'+self.ID+'.bg_par.dat','w') as file:
			file.write(str(self.bg_parameter))
		#print(self.bg_parameter)

		if plot:
			fig = plt.figure(figsize = (8, 8))
			grid = gridspec.GridSpec(2, 1, hspace=0.25)	
			freq = self.pds.f.values	
			ax1 = plt.subplot(grid[0,0])
			plt.plot(freq,self.pds.p, linewidth=0.5, color='black')
			plt.plot(freq,self.pds.bgfit, linewidth=1.0, color='blue' )
			plt.plot(freq,self.pds.fit, linewidth=1.0, color='red')
			plt.plot(freq,self.pds.lfit1, linestyle='dashed', linewidth=0.5, color='lime')
			plt.plot(freq,self.pds.lfit2, linestyle='dashed', linewidth=0.5, color='lime')
			plt.xscale('log')
			plt.yscale('log')
			plt.xlim(self.pds.f.min(),self.pds.f.max())
			plt.xlabel(r'Frequency ($\mu$Hz)')
			plt.ylabel(r'Power density (ppm$^2 / \mu$Hz)')
			plt.title(str(self.ID)+r': $\nu_\mathrm{max}$ = '+str(self.bg_parameter['fmax']) + r'    $p_\mathrm{powerexcess}$='+str(self.bg_parameter['probability']))

			ax1 = plt.subplot(grid[1,0])
			plt.plot(freq,self.pds.p, linewidth=0.5, color='black')
			plt.xlim(self.bg_parameter['fmax'].n-2.5*self.bg_parameter['sig'].n, self.bg_parameter['fmax'].n+2.5*self.bg_parameter['sig'].n)
			plt.xlabel(r'Frequency ($\mu$Hz)')
			plt.ylabel(r'Power density (ppm$^2 / \mu$Hz)')
			
			fig.savefig(self.path+'/'+self.ID+'.pdf', bbox_inches='tight')
			#plt.show()
			plt.close()
		print('-----> fmax fitting took me {}s'.format(round(time.time() - start_time,1)))

################################################################################################
	def dnu_estimator(self, plot=False, dnu_guess=False):
		if not dnu_guess:
			import pickle
			import zlib
			with open('RFmodel_dnu.pkl', 'rb') as file:
				model = pickle.load(file)
				rf = pickle.loads(zlib.decompress(model))
			X = []
			for key, value in self.bg_parameter.items():
				if key not in  ['n','probability']:
					X.append(value.n)
			predictions = rf.predict([X])[0]
			self.dnu_guess = predictions[0]
			self.dnu02_guess = predictions[1]
			self.dnu02_guess_err = 0.2
		else:
			self.dnu_guess = dnu_guess
			self.dnu02_guess = 0.08
			self.dnu02_guess_err = 0.5

		self.pds = self.pds[(self.pds.f > self.bg_parameter['fmax'].n - 2.5*self.dnu_guess) & (self.pds.f < self.bg_parameter['fmax'].n + 2.5*self.dnu_guess)].reset_index(drop=True)
		self.tau_guess = ufloat(77.9 - 21.6*np.log10(self.bg_parameter['fmax'].n),0)
		ag = 761.9 -613.9*np.log10(self.bg_parameter['fmax'].n) + 130.0*np.log10(self.bg_parameter['fmax'].n)**2
		#ag *= 5
		#print(predictions)
		pattern = self.LorModel(self.pds.f,
			self.bg_parameter['fmax'].n,
			self.dnu_guess,
			self.dnu02_guess,
			0,
			ag,ag/2,ag/3)
		f_cross, p_cross = self.crosscorr(self.pds.f,self.pds.p,pattern,[-0.8*self.dnu_guess,0.8*self.dnu_guess])
		p_cross *= np.exp(-f_cross**2/2/(self.dnu_guess/2)**2)
		f_offset = f_cross[np.argmax(p_cross)]
		self.fm_guess = self.bg_parameter['fmax'].n + f_offset

		pds_lw = self.pds[(self.pds.f > self.fm_guess-self.dnu_guess*self.dnu02_guess/2) & (self.pds.f < self.fm_guess+self.dnu_guess*self.dnu02_guess/2)].reset_index(drop=True)

		lp_func = lambda X, f, h, tau: pds_lw.bgfit + h / (1 + 4*((X-f)*np.pi*tau/11.57407)**2)    
		fit = NestedFitter(
			x=pds_lw.f.to_numpy(np.float32), 
			y=pds_lw.p.to_numpy(np.float32), 
			ye=[], 
			parameters=['f','h','lw'], 
			par_lo=[pds_lw.f.min(),0                 ,0.1*self.tau_guess.n], 
			par_hi=[pds_lw.f.max(),2.0*pds_lw.p.max(),3.0*self.tau_guess.n],
			function=lp_func,
			MLE=True,
			return_logZ=False,
			verbose=self.verbose)

		c,c_e = fit.fitter()

		self.fm_guess = c[0]
		self.height_guess = c[1]
		self.tau_guess = ufloat(c[2],c_e[2])

		if plot:
			lp_fit = lp_func(pds_lw.f.values, *c)
			fig = plt.figure()
			grid = gridspec.GridSpec(2, 1, hspace=0.25)	
			ax = plt.subplot(grid[0,0])
			plt.title(str(self.ID))
			plt.plot(self.pds.f,self.pds.p)
			plt.plot(self.pds.f,pattern,linewidth=0.5)
			plt.plot(self.pds.f,self.LorModel(self.pds.f,
				self.bg_parameter['fmax'].n + f_offset,
				self.dnu_guess,
				self.dnu02_guess,
				0,
				ag,ag/2,ag/3),linewidth=0.5)
			plt.plot(pds_lw.f,lp_fit)
			plt.axvline(self.fm_guess, color='red', linestyle='dashed', linewidth=0.5)
			ax = plt.subplot(grid[1,0])
			plt.plot(f_cross, p_cross)
			plt.axvline(f_offset, color='red', linestyle='dashed', linewidth=0.5)
			#plt.show()
			fig.savefig(self.path+'/'+self.ID+'.crosscor.pdf', bbox_inches='tight')
			plt.close()

################################################################################################
#####################         Fit Dnu model      ###############################################
################################################################################################
	def dnu_fitter(self, dnu_guess=False, flip_fc=False, plot=False):
		start_time = time.time()
		self.init_function(bg_par=True)

		#print(self.pds)
		#print(self.bg_parameter)
		pds_orig = self.pds.copy()	
		self.dnu_estimator(plot=plot, dnu_guess=dnu_guess)

		if flip_fc: self.fm_guess -= self.dnu_guess/2

		self.pds = self.pds[(self.pds.f > self.fm_guess - 1.5*self.dnu_guess) & (self.pds.f < self.fm_guess + 1.5*self.dnu_guess)].reset_index(drop=True)
		parameters=['f_c', 'dnu', 'dnu02', 'dnu01', 'h0', 'h1', 'h2']
		fit = NestedFitter(
			x=self.pds.f.to_numpy(np.float32), 
			y=self.pds.p.to_numpy(np.float32), 
			ye=[], 
			parameters=parameters, 
			par_lo=[self.fm_guess-self.dnu_guess/5,0.8*self.dnu_guess,(1-self.dnu02_guess_err)*self.dnu02_guess,-0.1,0,0,0], 
			par_hi=[self.fm_guess+self.dnu_guess/5,1.2*self.dnu_guess,(1+self.dnu02_guess_err)*self.dnu02_guess, 0.1,3*self.height_guess,3*self.height_guess,3*self.height_guess],
			function=self.LorModel,
			MLE=True,
			return_logZ=False,
			verbose=self.verbose)
		c,c_e = fit.fitter()


		self.pds['dnu_fit'] = self.LorModel(self.pds.f, *c)

		self.dnu_parameter = {}
		for i in range(len(parameters)):
			self.dnu_parameter[parameters[i]] = ufloat(c[i],c_e[i])

		eps = (c[0] / c[1]) % 1
		eps_e = np.sqrt(c_e[0]**2 * (1/c[1])**2 + c_e[1]**2 * (c[0]/c[1]**2)**2)
		eps_fit = 0.587+0.649*np.log10(c[1]) 
		if eps - eps_fit < -0.8: eps += 1
		eps_limit = -0.055
		xx = np.absolute(eps - eps_fit - eps_limit) / eps_e
		prob = 0.5 * (1 + math.erf(xx / np.sqrt(2)))

		### RGB
		if (eps - eps_fit > eps_limit) : evoflag = 0
		### 2nd clump
		if (eps - eps_fit < eps_limit) & (c[1] > 5) : evoflag = 2
		### RC
		if (eps - eps_fit < eps_limit) & (c[1] < 5) : evoflag = 1
		### AGB
		if (eps - eps_fit < eps_limit) & (c[1] < 3.3) : evoflag = 3
		### for MS and SG stars
		if self.bg_parameter['fmax'].n > 300: evoflag = 0
		
		evostr=['RGB','RC','2ndRC','AGB']
		#print(evostr[evoflag])
		#print(eps,eps_e,xx,prob)
		self.dnu_parameter['tau_guess'] = self.tau_guess
		self.dnu_parameter['eps'] = ufloat(eps,eps_e)
		self.dnu_parameter[evostr[evoflag]+'-prob'] = prob
		#print(self.dnu_parameter)
		
		with open(self.path+'/'+self.ID+'.dnu_par.dat','w') as file:
			file.write(str(self.dnu_parameter))

		if plot:
			fig = plt.figure(figsize = (8, 6))
			grid = gridspec.GridSpec(2, 1, hspace=0.25)	
			freq = pds_orig.f.values	
			ax1 = plt.subplot(grid[0,0])
			plt.plot(freq,pds_orig.p, linewidth=0.5, color='black')
			plt.plot(freq,pds_orig.bgfit, linewidth=1.0, color='blue' )
			plt.plot(freq,pds_orig.fit, linewidth=1.0, color='red')
			plt.plot(freq,pds_orig.lfit1, linestyle='dashed', linewidth=0.5, color='lime')
			plt.plot(freq,pds_orig.lfit2, linestyle='dashed', linewidth=0.5, color='lime')
			plt.xscale('log')
			plt.yscale('log')
			plt.xlim(pds_orig.f.min(),pds_orig.f.max())
			plt.xlabel(r'Frequency ($\mu$Hz)')
			plt.ylabel(r'Power density (ppm$^2 / \mu$Hz)')
			plt.title(str(self.ID)+r': $\nu_\mathrm{max}$ = '+str(self.bg_parameter['fmax']) + r'    $p_\mathrm{powerexcess}$='+str(self.bg_parameter['probability']))

			ax1 = plt.subplot(grid[1,0])
			plt.plot(self.pds.f,self.pds.p, linewidth=0.5, color='black')
			plt.plot(self.pds.f,self.pds.dnu_fit, linewidth=0.5, color='red')
			plt.axvline(self.bg_parameter['fmax'].n, color='blue', linestyle='dashed', linewidth=0.5)
			plt.axvline(self.dnu_parameter['f_c'].n, color='red', linestyle='dashed', linewidth=0.5)
			plt.xlabel(r'Frequency ($\mu$Hz)')
			plt.ylabel(r'Power density (ppm$^2 / \mu$Hz)')
			plt.title(r'$\Delta\nu$ = '+str(self.dnu_parameter['dnu'])+r'   $\nu_{0}$ = '+str(self.dnu_parameter['f_c'])+'  '+evostr[evoflag]+' ('+str(round(prob,2))+')')
			
			fig.savefig(self.path+'/'+self.ID+'.pdf', bbox_inches='tight')
			#plt.show()
			plt.close()


		#plt.plot(self.pds.f,self.pds.p)
		#plt.plot(self.pds.f,self.pds.dnu_fit)
		#plt.show()
		print('-----> dnu fitting took me {}s'.format(round(time.time() - start_time,1)))
###################################################################################################################
	def fit_doublemode(self, x, y, f0, dnu02, tau, mintau=None, log=False, rotation=False, incl_prior=None):
		f2 = f0 - dnu02
		if not mintau: mintau = 0.1*tau
		fit = NestedFitter(
			x=x, 
			y=y, 
			ye=[], 
			parameters=['f0', 'a0', 'tau0', 'f2', 'a2', 'tau2'], 
			par_lo=[f0-0.5*dnu02,0  ,mintau  ,f2-0.5*dnu02,0  ,mintau], 
			par_hi=[f0+0.5*dnu02,500,2.00*tau,f2+0.5*dnu02,500,2.00*tau],
			function=self.lor02,
			MLE=True,
			return_logZ=False,
			verbose=self.verbose)
		c,ce = fit.fitter()	

		if rotation:
			fit = NestedFitter(
				x=x, 
				y=y / self.lor(x,*c[:3]) * self.pds_now.bgfit, 
				ye=[], 
				parameters=['f2', 'a2', 'tau2', 'frot', 'incl'], 
				par_lo=[f2-0.5*dnu02,0  ,mintau  , 0, 0], 
				par_hi=[f2+0.5*dnu02,500,2.00*tau,0.5*dnu02,90],
				function=self.lor_rot,
				prior=incl_prior,
				MLE=True,
				return_logZ=False,
				verbose=self.verbose)
			c1,ce1 = fit.fitter()	
			c = np.concatenate((c[:3],c1))						
			ce = np.concatenate((ce[:3],ce1))						

		fit_1 = self.lor02_rot(x, *c) if rotation else self.lor02(x, *c)
		lgZ_1 = -np.sum(y/fit_1 + np.log(fit_1))
		cc = np.copy(c)
		cc[4] = 0 									### only l=0 mode
		fit_2 = self.lor02_rot(x, *cc) if rotation else self.lor02(x, *cc)
		lgZ_2 = -np.sum(y/fit_2 + np.log(fit_2))
		cc = np.copy(c)
		cc[1] = 0 									### only l=2 mode
		fit_3 = self.lor02_rot(x, *cc) if rotation else self.lor02(x, *cc)
		lgZ_3 = -np.sum(y/fit_3 + np.log(fit_3))
		lgZ_4 = - np.sum(y/self.pds_now.bgfit + np.log(self.pds_now.bgfit))
		lgZ = [lgZ_1, lgZ_2, lgZ_3, lgZ_4]
		if log:
			self.logger(u'fit parameter : f0={:.3f}\u00B1{:.3f} a0={:.1f} tau0={:.1f} f2={:.3f}\u00B1{:.3f} a2={:.1f} tau2={:.1f}'.format(c[0],ce[0],c[1],c[2],c[3],ce[3],c[4],c[5]))

		return c, ce, lgZ

################################################################################################
#####################         l=0,2 Peakbagging      ###########################################
################################################################################################
	def peakbag_02(self, alpha=None, l1_threshold=8, odds_ratio_limit = 5, mintau=None, rotation=False, incl_prior=None, plot=False, log=False):
		start_time = time.time()

		if incl_prior is not None: incl_prior.insert(0,-1)

		self.init_function(bg_par=True, dnu_par=True, freq=False)
		print(self.bg_parameter,self.dnu_parameter)

		orders = int(round(self.bg_parameter['sig'].n/self.dnu_parameter['dnu'].n))*3 + 1
		orders = list(np.arange(-orders,orders+1))
		print(orders)

		if not alpha:
			alpha = 0.043*self.dnu_parameter['dnu'].n**(-0.659) if self.evo == 'RGB' else 0.086*self.dnu_parameter['dnu'].n**(-1.086)
		print(alpha)

		dnu = self.dnu_parameter['dnu'].n
		dnu02 = self.dnu_parameter['dnu02'].n * dnu
		tau = self.dnu_parameter['tau_guess'].n
		self.freq = pd.DataFrame()
		
		self.pds['l02_fit'] = self.pds.bgfit

		if plot:
			nr = len(orders)
			if np.sqrt(nr) % 1 == 0:
				Nrows, Ncols = np.sqrt(nr), np.sqrt(nr)
			else:
				Ncols = round(np.sqrt(nr))
				Nrows = np.floor(nr/Ncols+1)
			fig = plt.figure(figsize = (10, 6))
			grid = gridspec.GridSpec(int(Nrows), int(Ncols), hspace=0.35)

		counter = 0
		for order in orders:
			f0 = self.dnu_parameter['f_c'].n + dnu * (order + alpha/2 * order**2)
			lim = np.abs(order) * 0.05 * dnu02
			self.pds_now = self.pds[(self.pds.f > f0 - 1.5*dnu02-lim) & (self.pds.f < f0 + 0.5*dnu02+lim)]

			if log: self.logger(u'---------- order: {} at {:.3f}\u00B1{:.3f} ---------'.format(order,f0,dnu02/2))


			c, ce, lgZ = self.fit_doublemode(
				self.pds_now.f.values, 
				self.pds_now.p.values, 
				f0, dnu02, tau, mintau=mintau, log=log, 
				rotation=rotation, incl_prior=incl_prior)
			p = self.evi2prob(lgZ)

			if l1_threshold:
				snr = self.pds_now.p/self.lor02(self.pds_now.f.values, *c)
				peaks, _ = find_peaks(snr, height=l1_threshold, distance=3)

				if len(peaks) > 0:
					if log: self.logger('{} dipole modes removed'.format(len(peaks)))
					xx = self.pds_now.f.to_numpy().copy()
					yy = self.pds_now.p.to_numpy().copy()
					zz = self.pds_now.bgfit.to_numpy().copy()
					for peak in peaks:
						yy[peak-2:peak+2] = zz[peak-2:peak+2]

					c, ce, lgZ = self.fit_doublemode(
						xx,
						yy, 
						f0, dnu02, tau, mintau=mintau, log=log,
						rotation=rotation, incl_prior=incl_prior)
					p = self.evi2prob(lgZ)
			odds_ratio_l0 = np.log((p[0] + p[1]) / (p[2] + p[3]) / odds_ratio_limit)
			odds_ratio_l2 = np.log((p[0] + p[2]) / (p[1] + p[3]) / odds_ratio_limit)
			if log: 
				self.logger('lgZ : S1={:.1f} S2={:.1f} S3={:.1f} S4={:.1f}'.format(*lgZ))
				self.logger('prob: S1={:.2f} S2={:.2f} S3={:.2f} S4={:.1f}'.format(*p))
				self.logger('ln(Odds ratio): l=0: {:.2f}  l=2: {:.2f}'.format(odds_ratio_l0,odds_ratio_l2))

			mode_selected = False
			if odds_ratio_l0 > 0:
				mode_selected = True
				if log: self.logger('l=0 mode is selected with p={:.2f}'.format(p[0]+p[1]))
				ff = pd.DataFrame({
						'l':[0], 
						'n_c':[order],
						'f': [c[0]], 'f_e': [ce[0]], 
						'a': [c[1]], 'a_e': [ce[1]],
						'tau':[c[2]], 'tau_e':[ce[2]],
						'p': [p[0]+p[1]]
						})
				if rotation:
					ff = pd.concat([ff, pd.DataFrame({
						'frot':[0], 'frot_e':[0],
						'incl':[0], 'incl_e':[0]
						})],axis=1)
				self.freq = pd.concat([self.freq, ff],ignore_index=True)
			if odds_ratio_l2 > 0:
				mode_selected = True
				if log: self.logger('l=2 mode is selected with p={:.2f}'.format(p[0]+p[3]))
				ff = pd.DataFrame({
						'l':[2], 
						'n_c':[order],
						'f': [c[3]], 'f_e': [ce[3]], 
						'a': [c[4]], 'a_e': [ce[4]],
						'tau':[c[5]], 'tau_e':[ce[5]],
						'p': [p[0]+p[2]]					
					})
				if rotation:
					ff = pd.concat([ff, pd.DataFrame({
						'frot':[c[6]], 'frot_e':[ce[6]],
						'incl':[c[7]], 'incl_e':[ce[7]]
						})],axis=1)
				self.freq = pd.concat([self.freq, ff],ignore_index=True)

			if mode_selected:
				fit = self.lor02_rot(self.pds.f, *c, use_bg=False) if rotation else self.lor02(self.pds.f, *c, use_bg=False)
				self.pds.l02_fit += fit

			else:
				print('no mode')
				if log: self.logger('no mode selected')

			if plot:
				fit = self.lor02_rot(self.pds_now.f, *c, use_bg=False) if rotation else self.lor02(self.pds_now.f, *c, use_bg=False)
				ax = plt.subplot(grid[int(np.floor(counter/Ncols)),int(counter % Ncols)])
				plt.plot(self.pds_now.f,self.pds_now.p,color='black',linewidth=0.5)
				plt.plot(self.pds_now.f,fit,color='red')
				if l1_threshold:
					if len(peaks) > 0:
						for peak in peaks:
							plt.axvline(self.pds_now.f.to_numpy()[peak],color='blue',linestyle='dashed',linewidth=0.5)
				#plt.title('p: l0:{:.2f} l2:{:.2f}'.format(p[0]+p[1],p[0]+p[2]))
				col = 'green' if odds_ratio_l0 > 0 else 'red'
				plt.text(0.05, 0.85, u'f0: {:.3f}\u00B1{:.3f} p={:.2f}'.format(c[0],ce[0],p[0]+p[1]), transform=ax.transAxes,color=col)
				col = 'green' if odds_ratio_l2 > 0 else 'red'
				plt.text(0.05, 0.75, u'f2: {:.3f}\u00B1{:.3f} p={:.2f}'.format(c[3],ce[3],p[0]+p[2]), transform=ax.transAxes,color=col)

			counter += 1

		self.pds.to_csv(self.path+'/'+self.ID+'.bg.fit.pds', index=False)

		dd = pd.DataFrame({
			'ID':[self.ID],
			'fmax':[self.bg_parameter['fmax'].n],
			'fmax_e':[self.bg_parameter['fmax'].s],
			'dnu':[self.dnu_parameter['dnu'].n],
			'dnu_e':[self.dnu_parameter['dnu'].s],
			'dnu02':[self.dnu_parameter['dnu02'].n],
			'dnu02_e':[self.dnu_parameter['dnu02'].s],
			'f0_c':[self.dnu_parameter['f_c'].n],
			'f0_c_e':[self.dnu_parameter['f_c'].s],
			'evo':list(self.dnu_parameter.keys())[-1].replace('-prob','')
			})

		self.freq = self.freq.reset_index(drop=True).sort_values(by=['l', 'f'],ignore_index=True)

		if plot:
			fig.savefig(self.path+'/'+self.ID+'.l02_modes.pdf', bbox_inches='tight')
			plt.close()

		#### compute curvature in l=0 modes
		freq0 = self.freq[self.freq.l == 0].reset_index(drop=True)
		cor_func = lambda X, f0, dnu, alpha: f0 + dnu*(X + 0.5*alpha*X**2)
		fit = NestedFitter(
			x=freq0.n_c.values, 
			y=freq0.f.values, 
			ye=freq0.f_e.values, 
			parameters=['f0', 'dnu', 'alpha'], 
			par_lo=[self.dnu_parameter['f_c'].n-self.dnu_parameter['dnu'].n/10,0.9*self.dnu_parameter['dnu'].n,0], 
			par_hi=[self.dnu_parameter['f_c'].n+self.dnu_parameter['dnu'].n/10,1.1*self.dnu_parameter['dnu'].n,3*alpha], 
			function=cor_func,
			verbose=self.verbose)

		curv_par,curv_par_e = fit.fitter()
		a_glitch = (np.std(freq0.f-cor_func(freq0.n_c,*curv_par))/curv_par[1]*100)
		curv = pd.DataFrame({
			'dnu_cor':[curv_par[1]],
			'dnu_cor_e':[curv_par_e[1]],
			'alpha':[curv_par[2]],
			'alpha_e':[curv_par_e[2]],
			'a_glitch':[a_glitch]
			})

		### save modes
		s = dd.to_string(index=False, float_format='%.3f')
		s += '\n\n'
		s += curv.to_string(index=False, float_format='%.4f')
		s += '\n\n'
		s += self.freq.to_string(index=False, float_format='%.3f')
		s += '\n'
		with open(self.path+'/'+self.ID+'.modes.dat', 'w') as ff:
			ff.write(s)	

		if plot: self.make_mode_plots(rotation=rotation)
		print('-----> l02 peakbagging took me {}s'.format(round(time.time() - start_time,1)))

###################################################################################################################
	def make_mode_plots(self, rotation=False):
		plt.rcParams.update({'font.size':4})

		dnu = self.dnu_parameter['dnu'].n
		f0 = self.freq[self.freq.l == 0].reset_index(drop=True)
		f1 = self.freq[self.freq.l == 1].reset_index(drop=True)
		f2 = self.freq[self.freq.l == 2].reset_index(drop=True)
		f3 = self.freq[self.freq.l == 3].reset_index(drop=True)
		frange = [self.freq.f.min()-0.5*dnu,self.freq.f.max()+0.5*dnu]
		n = len(f0)
		fig = plt.figure(figsize = (10, 6))
		grid = gridspec.GridSpec(n, 1, hspace=0.35)	if len(f1) == 0 else gridspec.GridSpec(n+1, 1, hspace=0.35)
		for i in range(n):
			ax = plt.subplot(grid[i,0])
			dd = self.pds[(self.pds.f > f0.loc[i,'f']-1.2*dnu) & (self.pds.f < f0.loc[i,'f']+0.2*dnu)].reset_index(drop=True)
			plt.plot(dd.f,dd.p,linewidth=0.5,color='black')
			plt.plot(dd.f,dd.l02_fit,linewidth=0.5,color='red')
			if 'l13_fit' in dd.columns:
				plt.plot(dd.f,dd.l13_fit,linewidth=0.5,color='blue')
			plt.plot(dd.f,8*dd.bgfit,linewidth=0.5,color='green',linestyle='dashed')
			plt.xlim(f0.loc[i,'f']-1.2*dnu,f0.loc[i,'f']+0.2*dnu)
			for j in range(len(f0)):
				plt.axvline(f0.loc[j,'f'],color='red',linestyle='dashed',linewidth=0.5)
			for j in range(len(f1)):
				plt.axvline(f1.loc[j,'f'],color='blue',linestyle='dashed',linewidth=0.5)
			for j in range(len(f2)):
				plt.axvline(f2.loc[j,'f'],color='orange',linestyle='dashed',linewidth=0.5)
			for j in range(len(f3)):
				plt.axvline(f3.loc[j,'f'],color='yellow',linestyle='dashed',linewidth=0.5)
			if i == 0: plt.title(self.ID)
		if len(f1) > 0:
			ax = plt.subplot(grid[i+1,0])
			plt.plot(self.pds.f,self.pds.p_res,linewidth=0.5,color='black')
			plt.plot(self.pds.f,8*self.pds.bgfit,linewidth=0.5,color='green',linestyle='dashed')

		plt.xlabel(r'$\nu$ ($\mu$Hz)')
		fig.savefig(self.path+'/'+self.ID+'.modes.pdf', bbox_inches='tight')
		#plt.show()
		plt.close()

		plt.rcParams.update({'font.size':6})
		fig = plt.figure(figsize = (6, 6))
		n = 4 if rotation else 3
		grid = gridspec.GridSpec(n, 1, hspace=0.25)	
		ax = plt.subplot(grid[0,0])	
		plt.plot(f0.f % dnu, f0.f,'o', markersize=6, markerfacecolor='red', markeredgewidth=1, markeredgecolor='black',label='l=0')
		plt.errorbar(f0.f % dnu, f0.f, yerr=0, xerr=f0.f_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		plt.plot(f2.f % dnu, f2.f,'o', markersize=6, markerfacecolor='orange', markeredgewidth=1, markeredgecolor='black',label='l=2')
		plt.errorbar(f2.f % dnu, f2.f, yerr=0, xerr=f2.f_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		if len(f1) > 0:
			plt.plot(f1.f % dnu, f1.f,'o', markersize=4, markerfacecolor='blue', markeredgewidth=1, markeredgecolor='black',label='l=1')
			plt.errorbar(f1.f % dnu, f1.f, yerr=0, xerr=f1.f_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		if len(f3) > 0:
			plt.plot(f3.f % dnu, f3.f,'o', markersize=4, markerfacecolor='yellow', markeredgewidth=1, markeredgecolor='black',label='l=3')
			plt.errorbar(f3.f % dnu, f3.f, yerr=0, xerr=f3.f_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		plt.legend()
		plt.xlim(0,dnu)
		plt.xlabel(r'$\nu$ mod $\Delta\nu$ ($\mu$Hz)'.format(round(dnu,3)))
		plt.ylabel(r'$\nu$ ($\mu$Hz)')
		#plt.title(r'{} -- $\Delta\nu$={} -- $\nu_\mathrm{max}$={}'.format(self.ID,self.dnu_parameter['dnu'],self.bg_parameter['fmax']))
		plt.title(str(self.ID)+r': $\nu_\mathrm{max}$ = '+str(self.bg_parameter['fmax']) + r'    $\Delta\nu$ = '+str(self.dnu_parameter['dnu']))

		ax = plt.subplot(grid[1,0])	
		plt.plot(f0.f, f0.a,'o', markersize=6, markerfacecolor='red', markeredgewidth=1, markeredgecolor='black')
		plt.errorbar(f0.f, f0.a, xerr=0, yerr=f0.a_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		plt.plot(f2.f, f2.a,'o', markersize=6, markerfacecolor='orange', markeredgewidth=1, markeredgecolor='black')
		plt.errorbar(f2.f, f2.a, xerr=0, yerr=f2.a_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		if len(f1) > 0:
			plt.plot(f1.f, f1.a,'o', markersize=4, markerfacecolor='blue', markeredgewidth=1, markeredgecolor='black')
			plt.errorbar(f1.f, f1.a, xerr=0, yerr=f1.a_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		if len(f3) > 0:
			plt.plot(f3.f, f3.a,'o', markersize=4, markerfacecolor='yellow', markeredgewidth=1, markeredgecolor='black')
			plt.errorbar(f3.f, f3.a, xerr=0, yerr=f3.a_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		plt.xlabel(r'$\nu$ ($\mu$Hz)')
		plt.ylabel(r'a (ppm)')
		plt.xlim(frange[0],frange[1])

		ax = plt.subplot(grid[2,0])	
		plt.plot(f0.f, f0.tau,'o', markersize=6, markerfacecolor='red', markeredgewidth=1, markeredgecolor='black')
		plt.errorbar(f0.f, f0.tau, xerr=0, yerr=f0.tau_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		plt.plot(f2.f, f2.tau,'o', markersize=6, markerfacecolor='orange', markeredgewidth=1, markeredgecolor='black')
		plt.errorbar(f2.f, f2.tau, xerr=0, yerr=f2.tau_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		if len(f1) > 0:
			plt.plot(f1.f, f1.tau,'o', markersize=4, markerfacecolor='blue', markeredgewidth=1, markeredgecolor='black')
			plt.errorbar(f1.f, f1.tau, xerr=0, yerr=f1.tau_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		if len(f3) > 0:
			plt.plot(f3.f, f3.tau,'o', markersize=4, markerfacecolor='yellow', markeredgewidth=1, markeredgecolor='black')
			plt.errorbar(f3.f, f3.tau, xerr=0, yerr=f3.tau_e, linewidth=0, elinewidth=0.5, label=None, color='black')
		plt.xlabel(r'$\nu$ ($\mu$Hz)')
		plt.ylabel(r'$\tau$ (d)')
		if len(f1) > 1:
			plt.yscale('log')
			plt.ylim(0.8*self.freq.tau.min(),1.2*self.freq.tau.max())
		plt.xlim(frange[0],frange[1])
		
		if rotation:
			ax = plt.subplot(grid[3,0])	
			if len(f1) > 0:
				plt.plot(f1.f, f1.frot,'o', markersize=6, markerfacecolor='blue', markeredgewidth=1, markeredgecolor='black')
				plt.errorbar(f1.f, f1.frot, xerr=0, yerr=f1.frot_e, linewidth=0, elinewidth=0.5, label=None, color='black')
			plt.plot(f2.f, f2.frot,'o', markersize=6, markerfacecolor='orange', markeredgewidth=1, markeredgecolor='black')
			plt.errorbar(f2.f, f2.frot, xerr=0, yerr=f2.frot_e, linewidth=0, elinewidth=0.5, label=None, color='black')
			plt.xlim(frange[0],frange[1])
			plt.xlabel(r'$\nu$ ($\mu$Hz)')
			plt.ylabel(r'$\delta\nu_\mathrm{rot}$ ($\mu$Hz)')

		fig.savefig(self.path+'/'+self.ID+'.echelle.pdf', bbox_inches='tight')
		#plt.show()
		plt.close()

################################################################################################
#####################         l=1,3 Peakbagging      ###########################################
################################################################################################
	def peakbag_13(self, ev_limit = 0.8, snr_limit=7.5, iter_limit=150, plot=False, log=False, maxtau=False, rotation=False, incl_prior=None):
		start_time = time.time()

		self.init_function(bg_par=True, dnu_par=True, freq=True)
		if 'l02_fit' not in self.pds.columns:
			print('you need to run l=0,2 mode peakbagging first')
			exit()
		if (self.evo == 'RGB') & (self.bg_parameter['fmax'].n < 30):
			print('not enough frequency resolution to extract mixed l=1 modes!!!')
			exit()
		if (self.evo == 'RC') & (self.bg_parameter['fmax'].n < 20):
			print('not enough frequency resolution to extract mixed l=1 modes!!!')
			exit()

		if incl_prior is not None: incl_prior.insert(0,-1)

		self.df = self.pds.loc[1,'f'] - self.pds.loc[0,'f']
		self.pds['snr'] = self.pds.p/self.pds.l02_fit
		self.pds['p_res'] = self.pds.p/self.pds.l02_fit*self.pds.bgfit
		dnu = self.dnu_parameter['dnu'].n 
		dnu02 = dnu * self.dnu_parameter['dnu02'].n
		self.pds = self.pds[(self.pds.f > self.freq[self.freq.l == 0].f.min()-dnu-dnu02) & (self.pds.f < self.freq[self.freq.l == 0].f.max()+dnu02)].reset_index(drop=True)
		print(self.bg_parameter, self.dnu_parameter)
		print(self.freq)


		if self.bg_parameter['fmax'].n < 300:
			### fitting window half width = 1st local minimum in autocorrelation of smoothed (snr) spectrum 
			sigma = 2*self.df
			gx = np.arange(-3*sigma, 3*sigma, self.df)
			kernel = np.exp(-(gx/sigma)**2/2)
			kde_vals = np.convolve(self.pds.snr.values, kernel, mode="same")
			x_cor, y_cor = self.crosscorr(self.pds.f,kde_vals,kde_vals,[0,self.dnu_parameter['dnu'].n/10])
			peaks, _ = find_peaks(1-y_cor**0.5, distance=10)
			win_hw = x_cor[peaks[0]]
			#plt.plot(x_cor,y_cor)
			#for p in peaks:
			#	plt.axvline(x_cor[p],linestyle='dashed')
			#plt.show()
			print('fitting window half width: {}'.format(win_hw))
		else: win_hw = dnu02/2
		#quit()
	
		self.pds['l13_fit'] = self.pds.bgfit
	
		self.freq_l1 = pd.DataFrame()
		iteration = 0
		if not maxtau: maxtau = round(11.57407 / (np.pi * self.df),-1)
		while True:
		#for iteration in range(50):
			idx = self.pds.snr.idxmax()
			f = self.pds.loc[idx,'f']
			f_snr = self.pds.loc[idx,'snr']
			print('max peak found at: {}'.format(f) )
			if log: 
				self.logger('------- {} -------'.format(iteration))
				self.logger(u'f={:.3f}\u00B1{:.3f} snr={:.1f}'.format(f,win_hw,f_snr))

			self.pds_now = self.pds[(self.pds.f > f - win_hw) & (self.pds.f < f + win_hw)].copy().reset_index(drop=True)
			maxamp = 3 * np.sqrt(self.df * self.pds_now.p_res.sum())
			lgZ_nosig = - np.sum(self.pds_now.p_res/self.pds_now.bgfit + np.log(self.pds_now.bgfit))

			if not rotation:
				Nfit = NestedFitter(
					x=self.pds_now.f.values, 
					y=self.pds_now.p_res.values, 
					ye=[], 
					parameters=['f', 'a', 'tau'], 
					par_lo=[f-5*self.df,0,50], 
					par_hi=[f+5*self.df,maxamp,maxtau],
					function=self.lor,
					MLE=True,
					return_logZ=True,
					verbose=self.verbose)
				c,ce,lgZ_lor = Nfit.fitter()
				fit = self.lor(self.pds.f,*c,use_bg=False) + self.pds.bgfit
			else:
				#prior = None if incl_prior is None else incl_prior.insert(0,4)
				Nfit = NestedFitter(
					x=self.pds_now.f.values, 
					y=self.pds_now.p_res.values, 
					ye=[], 
					parameters=['f', 'a', 'tau', 'frot', 'incl'], 
					par_lo=[f-dnu02/2,0,0,0,0], 
					par_hi=[f+dnu02/2,3*maxamp,maxtau,dnu02/4,90],
					function=self.lor_rot,
					prior=incl_prior,
					MLE=True,
					return_logZ=True,
					verbose=self.verbose)
				c,ce,lgZ_lor = Nfit.fitter()
				fit = self.lor_rot(self.pds.f,*c,use_bg=False) + self.pds.bgfit

			lw_df = 11.57407 / (np.pi * c[2]) / self.df
			p = self.evi2prob([lgZ_lor,lgZ_nosig])
			print(lw_df)
			if log: 
				self.logger('fit lor: f={:.3f} a={:.1f} tau={:.1f}'.format(*c))
				self.logger('fit lor: p1:{:.3f} p2:{:.3f}'.format(*p))

			if lw_df < 2:
				Nfit = NestedFitter(
					x=self.pds_now.f.values, 
					y=self.pds_now.p_res.values, 
					ye=[], 
					parameters=['f', 'a'], 
					par_lo=[f-5*self.df,0], 
					par_hi=[f+5*self.df,maxamp],
					function=self.sinc,
					MLE=True,
					return_logZ=True,
					verbose=self.verbose)

				c_sinc,ce_sinc,lgZ_sinc = Nfit.fitter()
				fit_sinc = self.sinc(self.pds.f,*c_sinc,use_bg=False) + self.pds.bgfit
				p_sinc = self.evi2prob([lgZ_lor,lgZ_sinc,lgZ_nosig])
				if log: 
					self.logger('fit sinc: f={:.3f} a={:.1f}'.format(*c))
					self.logger('fit sinc: p1:{:.3f} p2:{:.3f} p3:{:.3f}'.format(*p_sinc))
				print(p_sinc)
				if p_sinc[1] > 1 - ev_limit:		### -> prob of lor needs to be > 0.85
					c, ce = c_sinc, ce_sinc
					fit = fit_sinc
					c  = np.append(c,maxtau)
					ce = np.append(ce,0)
					p = p_sinc
			### prewithen the best fit
			self.pds.p_res = self.pds.p_res / fit * self.pds.bgfit
			self.pds.snr = self.pds.p_res/ fit

			idx = np.argmin(np.abs(f-self.freq.loc[self.freq.l == 0,'f'].to_numpy()))
			if f > self.freq.loc[idx,'f']: idx += 1
			n_c = self.freq.loc[idx,'n_c']

			if rotation:
				self.freq_l1 = pd.concat([self.freq_l1, pd.DataFrame({
					'l':[1], 
					'n_c': [n_c],
					'f': [c[0]], 'f_e': [ce[0]], 
					'a': [c[1]], 'a_e': [ce[1]],
					'tau':[c[2]], 'tau_e':[ce[2]],
					'frot':[c[3]], 'frot_e':[ce[3]],
					'incl':[c[4]], 'incl_e':[ce[4]],
					'p': [1-p[-1]],
					'snr': [f_snr]
					})],ignore_index=True)
			else:				
				self.freq_l1 = pd.concat([self.freq_l1, pd.DataFrame({
					'l':[1], 
					'n_c': [n_c],
					'f': [c[0]], 'f_e': [ce[0]], 
					'a': [c[1]], 'a_e': [ce[1]],
					'tau':[c[2]], 'tau_e':[ce[2]],
					'p': [1-p[-1]],
					'snr': [f_snr]
					})],ignore_index=True)

			self.pds.l13_fit += fit - self.pds.bgfit

			### stop criteria		
			iteration += 1
			if (iteration > iter_limit) or (p[-1] > 1-ev_limit) or (f_snr < snr_limit):
				break

		### identify l=3 modes
		f0mid = []
		freq0 = self.freq[self.freq.l == 0].reset_index(drop=True)
		for i in range(len(freq0)-1):
			f0mid.append((freq0.loc[i,'f']+freq0.loc[i+1,'f'])/2)

		for i in range(len(self.freq_l1)):
			idx = np.argmin(np.abs(self.freq_l1.loc[i,'f']-f0mid))
			self.freq_l1.at[i,'f0p'] = (self.freq_l1.loc[i,'f']-f0mid[idx])/dnu
			self.freq_l1.at[i,'a_rel'] = self.freq_l1.loc[i,'a'] / np.interp(self.freq_l1.loc[i,'f'],freq0.f,freq0.a) / (1/(1+4*(self.freq_l1.loc[i,'f0p']/0.08)**2) + 0.05)

		factor = 0 if self.bg_parameter['fmax'].n > 300 else 0.0022*dnu
		l3_range = [-0.31-0.05 + factor, -0.31+0.05 + factor]
		idx = self.freq_l1[(self.freq_l1.f0p > l3_range[0]) & (self.freq_l1.f0p < l3_range[1]) & (self.freq_l1.a_rel > 2) & (self.freq_l1.tau < 200)].index.to_list()
		if len(idx) > 0: 
			for i in idx:
				self.freq_l1.at[i,'l'] = 3
		self.freq_l1 = self.freq_l1.drop(['f0p','a_rel'],axis=1)

		### combine old freq dataframe with new one
		self.freq = self.freq[(self.freq.l != 1) & (self.freq.l < 3)].reset_index(drop=True)
		self.freq['snr'] = 0
		if rotation:
			if 'frot' not in self.freq.columns:
				self.freq['frot'] = 0
				self.freq['frot_e'] = 0
				self.freq['incl'] = 0
				self.freq['incl_e'] = 0
		self.freq = pd.concat([self.freq,self.freq_l1], ignore_index=True)
		self.freq = self.freq.sort_values(by=['l', 'f'],ignore_index=True)

		### save modes
		s = ''
		for ss in self.freq_header: s += ss
		s += self.freq.to_string(index=False, float_format='%.3f')
		s += '\n'
		with open(self.path+'/'+self.ID+'.modes.dat', 'w') as ff:
			ff.write(s)	

		#plt.plot(self.pds.f,self.pds.p_res)
		#plt.plot(self.pds.f,8*self.pds.bgfit)
		#plt.show()

		if plot: self.make_mode_plots(rotation=rotation)
		print('-----> l13 peakbagging took me {}s'.format(round(time.time() - start_time,1)))

