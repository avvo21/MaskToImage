import torch
import torch.optim as optim
import torch.nn as nn
import os
import torchvision
import torch.nn.functional as f
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from model import UNetGenerator, NLayerDiscriminator
from tg_bot import TelegramBot
from utils import calculate_psnr, calculate_ssim, decode_segmap


def init_weights(m):
    classname = m.__class__.__name__ # Get the class name of the layer

    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02) # Common practice for GANs
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        # For batch norm layers, the weight is initialized to a normal distribution
        # with mean=1.0 and std=0.02 to start close to an identity transformation.
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class Pix2PixSolver(object):
    def __init__(self, train_loader, test_loader, device, writer, args):
        self.args = args
        self.model_name = 'Mask2Img_Pix2Pix_{}.pth'.format(self.args.model_name)
        self.device = device
        self.writer = writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = args.epochs

        # Models
        self.netG = UNetGenerator(args).to(device)
        self.netD = NLayerDiscriminator(input_nc=22, ndf=64, n_layers=args.n_layer_d, norm_type=args.norm_type).to(device)

        # Weight Initialization
        self.netG.apply(init_weights)
        self.netD.apply(init_weights)

        if self.args.resume_train:
            self.load_models()

        # Losses and Optimizers
        if args.gan_mode == 'vanilla':
            self.criterionGAN = nn.BCEWithLogitsLoss() # Standard GAN loss
        elif args.gan_mode == 'lsgan':
            self.criterionGAN = nn.MSELoss() # Least Squares GAN loss

        self.criterionL1 = nn.L1Loss()
        self.lambda_l1 = args.lambda_l1

        # Define the optimizers
        self.optimizer_G = optim.Adam(self.netG.parameters(), lr=args.lr_g, betas=(args.beta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=args.lr_d, betas=(args.beta1, 0.999))

        # GradScaler manage the scaling of gradients to prevent underflow in FP16
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

        # Metrics and Bot
        self.best_fid = float('inf')
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

        # Telegram Bot Integration
        if args.telegram_token and args.telegram_chat_id:
            self.tg_bot = TelegramBot(args.telegram_token, args.telegram_chat_id)
        else:
            self.tg_bot = None

    def _get_one_hot_mask(self, mask):

        mask_clean = mask.clone()
        mask_void = (mask == 255) # Identify void pixels

        mask_clean[mask_void] = 0  # Temporary to avoid out-of-bounds in one_hot
        one_hot = f.one_hot(mask_clean, num_classes=19).permute(0, 3, 1, 2).float()

        # Zero out all channels where there was void
        one_hot.masked_fill_(mask_void.unsqueeze(1).to(self.device), 0.0)

        return one_hot.to(self.device)

    def _set_eval_mode(self): # Helper function for use dropout in eval mode
        self.netG.eval()
        # In the original Pix2Pix paper, dropout is kept active even during generation
        if getattr(self.args, 'dropout_mode', 'full') == 'paper' and self.args.dropout > 0:

            def enable_dropout(m): # Check if m is a dropout layer and call m.train on it
                if isinstance(m, torch.nn.Dropout):
                    m.train() # It enables the dropout

            self.netG.apply(enable_dropout)


    def train(self):
        print(f"Starting {self.model_name} Training...")
        iter_per_epoch = len(self.train_loader)

        if self.tg_bot:
            self.tg_bot.send_message(f"{self.model_name}: Training Started...")

        for epoch in range(self.epochs):
            # Both are set in training mode
            self.netG.train()
            self.netD.train()

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}") # For update bar

            for i, (masks, real_imgs) in enumerate(loop):

                # Move data to device
                masks, real_imgs= masks.to(self.device), real_imgs.to(self.device)

                masks_one_hot = self._get_one_hot_mask(masks) # Convert the mask to one-hot format for D

                # --- Update Discriminator ----------------------------------------------------------------------------
                self.optimizer_D.zero_grad() # Zeros the gradient of D

                # Autocast manages mixed precision during the forward pass
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):

                    fake_imgs = self.netG(masks) # Generate the fake images

                    # Fake
                    fake_cat = torch.cat((masks_one_hot, fake_imgs.detach()), 1)
                    pred_fake = self.netD(fake_cat)
                    # They should all be predicted as fake
                    loss_D_fake = self.criterionGAN(pred_fake, torch.zeros_like(pred_fake))

                    # Real
                    real_cat = torch.cat((masks_one_hot, real_imgs), 1)
                    pred_real = self.netD(real_cat)
                    # They should all be predicted as real
                    loss_D_real = self.criterionGAN(pred_real, torch.ones_like(pred_real))

                    loss_D = (loss_D_real + loss_D_fake) * 0.5 # Merge the two losses

                self.scaler.scale(loss_D).backward() # Scale the loss and backpropagate
                # Resets accumulated gradients to their original scale and update the weights
                self.scaler.step(self.optimizer_D)

                # --- Update Generator --------------------------------------------------------------------------------
                self.optimizer_G.zero_grad() # Zeros the gradient of G

                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):

                    fake_cat_for_G = torch.cat((masks_one_hot, fake_imgs), 1)
                    pred_fake_G = self.netD(fake_cat_for_G) # produces a prediction about how real the image appears

                    # The G target is all images predicted as real
                    loss_G_GAN = self.criterionGAN(pred_fake_G, torch.ones_like(pred_fake_G))
                    loss_G_L1 = self.criterionL1(fake_imgs, real_imgs)

                    loss_G = loss_G_GAN + (loss_G_L1 * self.lambda_l1) # Merge the losses with lambda factor

                self.scaler.scale(loss_G).backward() # Scale the loss and backpropagate
                # Resets accumulated gradients to their original scale and update the weights
                self.scaler.step(self.optimizer_G)

                self.scaler.update() # Update the scaler state

                # TensorBoard log
                if i % self.args.print_every == 0:
                    current_step = epoch * iter_per_epoch + i
                    self.writer.add_scalar('Loss/D_total', loss_D.item(), current_step)
                    self.writer.add_scalar('Loss/G_total', loss_G.item(), current_step)
                    self.writer.add_scalar('Loss/G_L1', loss_G_L1.item(), current_step)
                    self.writer.add_scalar('Loss/G_GAN', loss_G_GAN.item(), current_step)

            self.save_models("last")
            self.validate(epoch)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.epochs: # Compute the FID evry 10 epoch or at the last one
                current_fid = self.compute_fid()

                if self.tg_bot is not None:
                    self.tg_bot.send_message(f"FID at epoch {epoch + 1}: {current_fid}")

                self.writer.add_scalar('Val/FID', current_fid, epoch + 1) # Log for TensorBoard

                if current_fid < self.best_fid:
                    self.best_fid = current_fid
                    self.save_models("best_fid")

    def validate(self, epoch):

        self._set_eval_mode() # Set the model in eval mode (not using .eval for pix2pix dropout in generation)

        val_l1, val_psnr, val_ssim = 0.0, 0.0, 0.0

        with torch.no_grad(): # Disable gradient calculation

            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):

                for i, (masks, real_imgs) in enumerate(self.test_loader):

                    masks, real_imgs = masks.to(self.device), real_imgs.to(self.device)
                    fake_imgs = self.netG(masks) # Generate the fake images

                    # Compute all metrics
                    val_l1 += self.criterionL1(fake_imgs, real_imgs).item()
                    val_psnr += calculate_psnr(fake_imgs, real_imgs)
                    val_ssim += calculate_ssim(fake_imgs, real_imgs)

                    if i == 0:
                        # Convert the first 4 masks into a visual mask
                        mask_visual = decode_segmap(masks[:4], self.device)
                        # Concat mask, fake image and real image for visualization
                        comparison = torch.cat([mask_visual, fake_imgs[:4], real_imgs[:4]], dim=3)
                        # Generate the grid
                        grid = torchvision.utils.make_grid(comparison, nrow=1, normalize=True, value_range=(-1, 1))
                        self.writer.add_image('Val/Comparison_Input_Output_GT', grid, epoch) # TensorBoard Log

        # TensorBoard Log
        avg_l1 = val_l1 / len(self.test_loader)
        self.writer.add_scalar('Val/L1', avg_l1, epoch + 1)
        self.writer.add_scalar('Val/PSNR', val_psnr / len(self.test_loader), epoch + 1)
        self.writer.add_scalar('Val/SSIM', val_ssim / len(self.test_loader), epoch + 1)

        if self.tg_bot:
            self.tg_bot.send_message(f"Epoch {epoch + 1} Done. Val L1: {avg_l1:.4f}")
        self.netG.train()

    def compute_fid(self):

        self._set_eval_mode() # Set the model in eval mode (not using .eval for pix2pix dropout in generation)
        self.fid_metric.reset() # Reset the FID state

        with torch.no_grad(): # Disable gradient calculation
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):

                for masks, real_imgs in tqdm(self.test_loader, desc="FID"): # Iterate on test_loader
                    masks, real_imgs = masks.to(self.device), real_imgs.to(self.device)
                    fake_imgs = self.netG(masks) # Generate the fake images

                    # Conversion for FID (uint8 [0, 255])
                    real_b = ((real_imgs + 1) / 2).clamp(0, 1)
                    fake_b = ((fake_imgs + 1) / 2).clamp(0, 1)

                    # Update FID metrics
                    self.fid_metric.update((real_b * 255).to(torch.uint8), real=True)
                    self.fid_metric.update((fake_b * 255).to(torch.uint8), real=False)

        fid_score = self.fid_metric.compute().item() # Compute de final FID
        self.netG.train() # Set the model back to train mode

        return fid_score

    def save_models(self, suffix):
        path_g = os.path.join(self.args.checkpoint_path, f"{self.args.run_name}_G_{suffix}.pth")
        path_d = os.path.join(self.args.checkpoint_path, f"{self.args.run_name}_D_{suffix}.pth")
        torch.save(self.netG.state_dict(), path_g)
        torch.save(self.netD.state_dict(), path_d)

    def load_models(self):
        path_g = os.path.join(self.args.checkpoint_path, f"{self.args.run_name}_G_last.pth")
        path_d = os.path.join(self.args.checkpoint_path, f"{self.args.run_name}_D_last.pth")
        if os.path.exists(path_g):
            try:
                self.netG.load_state_dict(torch.load(path_g, map_location=self.device))
                self.netD.load_state_dict(torch.load(path_d, map_location=self.device))
                print(f"Models loaded from {path_g}")
            except Exception as e:
                print(f"Error loading models: {e}")