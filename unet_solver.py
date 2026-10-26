import torch
import torch.optim as optim
import torch.nn as nn
import os
import torchvision
from model import UNetGenerator
from tg_bot import TelegramBot
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from utils import calculate_psnr, calculate_ssim, decode_segmap

def init_weights(m):
    classname = m.__class__.__name__

    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02) # Common practice for generators
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

    elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
        # For batch norm layers, the weight is initialized to a normal distribution
        # with mean=1.0 and std=0.02 to start close to an identity transformation.
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


class UNetSolver(object):
    def __init__(self, train_loader, test_loader, device, writer, args):
        self.args = args
        self.model_name = 'Mask2Img_{}.pth'.format(self.args.model_name)
        self.device = device
        self.writer = writer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = self.args.epochs


        self.net = UNetGenerator(self.args).to(device) # Initializing the model
        self.net.apply(init_weights) # Weight initialization

        if self.args.resume_train:
            self.load_model()

        # Loss configuration
        if args.loss_function == 'L1':
            self.criterion = nn.L1Loss()
        elif self.args.loss_function == 'MSE':
            self.criterion = nn.MSELoss()

        # Optimizer configuration
        if self.args.opt == "SGD":
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr, momentum=0.9)
        elif self.args.opt == "Adam":
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, betas=(0.5, 0.999))

        # GradScaler manage the scaling of gradients to prevent underflow in FP16
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.device.type == 'cuda'))

        # Metrics and Bot
        self.best_val_loss = float('inf')
        self.fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(self.device)

        # Telegram Bot Integration
        if args.telegram_token and args.telegram_chat_id:
            self.tg_bot = TelegramBot(args.telegram_token, args.telegram_chat_id)
        else:
            self.tg_bot = None

    def train(self):

        print(f"Training Started (Norm: {self.args.norm_type}, Feat: {self.args.feat})...")
        if self.tg_bot:
            self.tg_bot.send_message(f"{self.model_name}: Training Started...")

        iter_per_epoch = len(self.train_loader)

        for epoch in range(self.epochs):

            self.net.train()

            running_loss = 0.0
            loop = tqdm(self.train_loader, total=iter_per_epoch, desc=f"Epoch {epoch + 1}/{self.epochs}") # For update bar

            for i, (masks, images) in enumerate(loop):

                # Move data to device
                masks, images = masks.to(self.device), images.to(self.device)

                self.optimizer.zero_grad() # Zeros the gradient

                # Autocast manages mixed precision during the forward pass
                with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):

                    outputs = self.net(masks) # Generate the fake images
                    loss = self.criterion(outputs, images) # Compute the loss

                self.scaler.scale(loss).backward() # Scale the loss and backpropagate
                # Resets accumulated gradients to their original scale and update the weights
                self.scaler.step(self.optimizer)
                self.scaler.update() # Update the scaler state

                running_loss += loss.item() # Cumulate the loss value

                # TensorBoard log
                if i % self.args.print_every == 0:
                    current_step = epoch * iter_per_epoch + i
                    self.writer.add_scalar('Loss/Train', loss.item(), current_step)

            self.validate(epoch) # Compute the validation metrics
            self.save_model(suffix="last")

        # Final FID calculation
        final_fid = self.compute_fid()
        self.writer.add_scalar('Final_FID', final_fid, self.epochs)
        if self.tg_bot:
            self.tg_bot.send_message(f"Training Complete! Final FID: {final_fid:.4f}")

        self.writer.close()

    def validate(self, epoch):

        self.net.eval() # Set the model in eval mode

        val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):

                for i, (masks, images) in enumerate(self.test_loader):

                    masks, images = masks.to(self.device), images.to(self.device)
                    outputs = self.net(masks) # Generate the fake images

                    # Compute all metrics
                    val_loss += self.criterion(outputs, images).item()
                    val_psnr += calculate_psnr(outputs, images)
                    val_ssim += calculate_ssim(outputs, images)

                    if i == 0:
                        mask_visual = decode_segmap(masks[:4], self.device)
                        comparison = torch.cat([mask_visual, outputs[:4], images[:4]], dim=3)
                        grid = torchvision.utils.make_grid(comparison, nrow=1, normalize=True, value_range=(-1, 1))
                        self.writer.add_image('Val/Comparison_Input_Output_GT', grid, epoch)

        # TensorBoard Logs
        avg_loss = val_loss / len(self.test_loader)
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/PSNR', val_psnr / len(self.test_loader), epoch)
        self.writer.add_scalar('Val/SSIM', val_ssim / len(self.test_loader), epoch)

        if self.tg_bot:
            self.tg_bot.send_message(f"Epoch {epoch + 1} Done. Val Loss: {avg_loss:.4f}")

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model(suffix="best")
            print(f"New best model: {avg_loss:.4f}")


    def compute_fid(self):

        self.net.eval() # Set the model in eval mode
        self.fid_metric.reset() # Reset the FID state

        with torch.no_grad(): # Disable gradient calculation
            with torch.amp.autocast('cuda', enabled=(self.device.type == 'cuda')):

                for masks, images in tqdm(self.test_loader, desc="FID"): # Iterate on test_loader
                    masks, images = masks.to(self.device), images.to(self.device)
                    fakes = self.net(masks) # Generate the fake images

                    # Normalization for FID
                    real_uint8 = ((images + 1) / 2).clamp(0, 1) * 255
                    fake_uint8 = ((fakes + 1) / 2).clamp(0, 1) * 255

                    # Update FID metrics
                    self.fid_metric.update(real_uint8.to(torch.uint8), real=True)
                    self.fid_metric.update(fake_uint8.to(torch.uint8), real=False)

        return self.fid_metric.compute().item()

    def save_model(self, suffix=""):
        check_path = os.path.join(self.args.checkpoint_path, f'Mask2Img_{self.args.model_name}_{suffix}.pth')
        torch.save(self.net.state_dict(), check_path)

    def load_model(self):
        check_path = os.path.join(self.args.checkpoint_path, f'Mask2Img_{self.args.model_name}_last.pth')
        if os.path.exists(check_path):
            self.net.load_state_dict(torch.load(check_path, map_location=self.device))
            print("Model loaded successfully!")