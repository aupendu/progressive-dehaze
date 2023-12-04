import torch
import time
from decimal import Decimal
import torchvision

from common import Image_Quality_Metric

class Trainer():
	def __init__(self, args, my_loader=None, my_modelt=None, my_modela=None, my_modelh=None, my_loss=None):
		self.args = args
		self.loader_train, self.loader_valid = my_loader.loader_train, my_loader.loader_valid
		self.trainfolder = args.trainfolder
		self.lossname = args.loss
		if args.train_h:
			self.modeltrans, self.modelatm, self.model = my_modelt.cuda(), my_modela.cuda(), my_modelh.cuda()
			self.model_name = args.hazemodel
			self.tr_type='Hazy Map:   '
		else:
			if args.train_t: 
				self.model = my_modelt.cuda()
				self.model_name = args.transmodel
				self.tr_type='Transmission Map:   '
			if args.train_a: 
				self.model = my_modela.cuda()
				self.model_name = args.atmmodel
				self.tr_type='Atmospheric Map:   '

		self.decay = args.decay.split('+')
		self.decay = [int(i) for i in self.decay] 
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.decay, gamma=args.gamma)
		self.loss = my_loss
		self.quality = Image_Quality_Metric()
		self.bestvalssim, self.bestvalpsnr, self.bestvalmse = 0, 0, 1e8
		self.bestepochS, self.bestepochP, self.bestepochM = 0, 0, 0
		if args.train_t or args.train_a:
			if args.resume:
				if args.train_t:
					checkpoint = torch.load(args.transmodel_pt)
					self.bestvalssim = checkpoint['ssim']
					self.bestvalpsnr = checkpoint['psnr']
					self.bestepochS = checkpoint['epochS']
					self.bestepochP = checkpoint['epochP']
					print(self.bestepochS, self.bestvalssim)
				if args.train_a:
					checkpoint = torch.load(args.atmmodel_pt)
					self.bestvalmse = checkpoint['mse']
					self.bestepochM = checkpoint['epochM']
					print(self.bestepochM, float(self.bestvalmse))
				self.model.load_state_dict(checkpoint['model'])
				self.optimizer.load_state_dict(checkpoint['optimizer'])
				self.scheduler.load_state_dict(checkpoint['scheduler'])

		if args.train_h:
			if args.resume:
				checkpoint = torch.load(args.hazemodel_pt)
				self.model.load_state_dict(checkpoint['model'])
				if args.resume:
					self.optimizer.load_state_dict(checkpoint['optimizer'])
					self.scheduler.load_state_dict(checkpoint['scheduler'])
					self.bestvalpsnr = checkpoint['psnr']
					self.bestepochP = checkpoint['epochP']
					print(self.scheduler.last_epoch, self.bestvalpsnr)
			
			checkpoint = torch.load(args.transmodel_pt)
			self.modeltrans.load_state_dict(checkpoint['model'])
			checkpoint = torch.load(args.atmmodel_pt)
			self.modelatm.load_state_dict(checkpoint['model'])
			
	def train(self):
		self.model.train()
		if self.args.train_h:
			self.modelatm.eval()
			self.modeltrans.eval()
		train_loss = 0
		lr = self.optimizer.param_groups[0]['lr']
		print('===========================================\n')
		print(self.tr_type+self.model_name+'   '+self.args.exp_name)
		print('[Epoch: %d] [Learning Rate: %.2e]'%(self.scheduler.last_epoch+1, Decimal(lr)))
		startepoch = time.time()
		for i_batch, (Cleanbatch, Hazybatch, Transbatch, Abatch) in enumerate(self.loader_train):
			if self.args.train_h:
				with torch.no_grad():
					trans_map = self.modeltrans((Hazybatch/255.).cuda())
					atm_map = self.modelatm((Hazybatch/255.).cuda())
					atm_map = torch.unsqueeze(torch.unsqueeze(atm_map, 2), 2)
					atm_map = atm_map.expand_as(Hazybatch)
				trans_map, atm_map = trans_map.detach(), atm_map.detach()
				Input, Output = Hazybatch.cuda(), Cleanbatch.cuda()
			if self.args.train_t: Input, Output = Hazybatch.cuda(), Transbatch.cuda()
			if self.args.train_a: Input, Output = Hazybatch.cuda(), Abatch.cuda()

			self.optimizer.zero_grad()
			if self.args.train_h:
				iOutput = self.model((Input/255., trans_map, atm_map, False))
				loss = self.loss(iOutput, Output/255.)
			else:
				iOutput = self.model(Input/255.)
				if self.args.train_a: Output = torch.squeeze(Output)
				loss = self.loss(iOutput, Output)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.data.cpu()

			if (i_batch+1)%100==0:
				train_loss = train_loss/100
				total_time = time.time()-startepoch
				startepoch = time.time()
				print('[Batch: %d] [Train Loss: %f] [Time: %.2f s]'\
					%(i_batch+1, train_loss, total_time))
				train_loss = 0
		self.scheduler.step()

	def valid(self):
		torch.set_grad_enabled(False)
		self.model.eval()
		if self.args.train_h:
			self.modelatm.eval()
			self.modeltrans.eval()
		val_mse = 0
		val_ssim = 0
		val_psnr = 0
		for i_batch, (CleanImage, HazyImage, TransImage, AtmImage) in enumerate(self.loader_valid):
			if self.args.train_t: Input, Output = HazyImage.cuda(), TransImage.cuda()
			if self.args.train_a: Input, Output = HazyImage.cuda(), AtmImage.cuda()
			torch.cuda.empty_cache()
			with torch.no_grad():
				if not self.args.train_h:
					iOutput = self.model(Input/255.)
				else:
					HazyImage, CleanImage = (HazyImage.cuda())/255., (CleanImage.cuda())/255.
					atm_map = self.modelatm(HazyImage)
					trans_map = self.modeltrans(HazyImage)
					atm_map = torch.unsqueeze(torch.unsqueeze(atm_map, 2), 2)
					atm_map = atm_map.expand_as(HazyImage)

					HazefreeImage = self.model((HazyImage, trans_map, atm_map, False))
					HazefreeImage = torch.round(255*torch.clamp(HazefreeImage, 0, 1))
					val_psnr += self.quality.psnr(HazefreeImage/255.0, CleanImage, rgb_range=1)
			if self.args.train_t:
				val_ssim += self.quality.ssim(iOutput, Output, rgb_range=1)
				val_psnr += self.quality.psnr(iOutput, Output, rgb_range=1)
			if self.args.train_a:
				Output = torch.squeeze(Output)
				val_mse += self.quality.mse(iOutput, Output)
		torch.set_grad_enabled(True)

		if self.args.train_a:
			tempmse = val_mse/len(self.loader_valid)
			if self.bestvalmse>tempmse:
				self.bestepochM = self.scheduler.last_epoch
				self.bestvalmse = tempmse
				self.save_models('model_dir/'+self.args.exp_name+'_'+self.model_name+'_'+self.lossname+'_bestMSE.pth.tar')
			self.save_models('model_dir/'+self.args.exp_name+'_'+self.model_name+'_'+self.lossname+'_check.pth.tar')
			print('\n\nValidation MSE:     %f     (Best MSE: %f at %d)'%(tempmse, self.bestvalmse, self.bestepochM))

		if self.args.train_h:
			temppsnr = val_psnr/len(self.loader_valid)
			if self.bestvalpsnr<temppsnr:
				self.bestepochP = self.scheduler.last_epoch
				self.bestvalpsnr = temppsnr
				self.save_models('model_dir/'+'I'+str(self.args.iter)+self.args.exp_name+'_'+self.model_name+'_'+self.lossname+'_bestPSNR.pth.tar')
			self.save_models('model_dir/'+'I'+str(self.args.iter)+self.args.exp_name+'_'+self.model_name+'_'+self.lossname+'_check.pth.tar')
			print('\n\nValidation PSNR:    %.2f    (Best PSNR: %.2f at %d)'%(temppsnr, self.bestvalpsnr, self.bestepochP))

		if self.args.train_t:
			temppsnr = val_psnr/len(self.loader_valid)
			tempssim = val_ssim/len(self.loader_valid)
			if self.bestvalpsnr<temppsnr:
				self.bestepochP = self.scheduler.last_epoch
				self.bestvalpsnr = temppsnr
				self.save_models('model_dir/'+self.args.exp_name+'_'+self.model_name+'_'+self.lossname+'_bestPSNR.pth.tar')
			print('\n\nValidation PSNR:    %.2f    (Best PSNR: %.2f at %d)'%(temppsnr, self.bestvalpsnr, self.bestepochP))
			if self.bestvalssim<tempssim:
				self.bestepochS = self.scheduler.last_epoch
				self.bestvalssim = tempssim
				self.save_models('model_dir/'+self.args.exp_name+'_'+self.model_name+'_'+self.lossname+'_bestSSIM.pth.tar')
			self.save_models('model_dir/'+self.args.exp_name+'_'+self.model_name+'_'+self.lossname+'_check.pth.tar')
			print('Validation SSIM:    %.3f    (Best SSIM: %.3f at %d)'%(tempssim, self.bestvalssim, self.bestepochS))


	def save_models(self, iname):
		torch.save({'epochP': self.bestepochP, 'epochS': self.bestepochS, 'epochM': self.bestepochM, 
					'model': self.model.state_dict(),
					'optimizer': self.optimizer.state_dict(), 
					'scheduler': self.scheduler.state_dict(),
					'psnr': self.bestvalpsnr, 'ssim': self.bestvalssim, 'mse': self.bestvalmse,}, iname)
		return None

	def terminate(self):
		epoch = self.scheduler.last_epoch
		return epoch >= self.args.epochs
