import random

import numpy as np
import torch
from torchvision.transforms import PILToTensor
from PIL import Image, ImageEnhance, ImageOps

class Randbox(object):
    def __init__(self, p1, alpha):
        self.p1 = p1
        self.alpha = alpha

    def generate_rand_mask(self, image):
        h, w, _ = image.shape
        mask = torch.zeros(h, w)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length = np.random.randint(int(np.sqrt(4 * h)), h)
        width  = np.random.randint(int(np.sqrt(4 * w)), w)
        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - width // 2, 0, w)
        x2 = np.clip(x + width // 2, 0, w)
        mask[y1: y2, x1: x2] = 1.
        return mask

    def get_segment_noise(self, image, alpha=1.0):
        h, w, c = image.shape
        image_tensor = torch.from_numpy(image)
        flag = False
        mask_noise_pil = None
        while not flag:
            mask = self.generate_rand_mask(image)
            mask_expand = mask.expand(3, -1, -1).permute(1, 2, 0)
            masked_image = torch.multiply(image_tensor, mask_expand)
            noise_image_tensor = torch.zeros([h,w,4])
            for c in range(3):
                tmp_channel = masked_image[:, :, c]
                # x = tmp_channel[tmp_channel != 0]
                x = tmp_channel
                if len(x) > 0:
                    x = x.double()
                    std, mean = torch.std_mean(x)
                    try:
                        normal_beta = torch.distributions.Normal(loc=mean, scale=std)
                        beta = normal_beta.sample([h * w]).reshape(h, w)
                        noise_image_tensor[:, :, c] = beta
                        flag = True
                    except:
                        # print(f"x={x} mean={mean}, std={std}")
                        flag = False
                        break
            if flag:        
                noise_image_alpha = torch.multiply(mask.squeeze(), alpha)
                noise_image_tensor[:, :, 3] = noise_image_alpha
                mask_expand2 = mask.expand(4, -1, -1).permute(1, 2, 0)
                noise_masked_tensor = torch.multiply(noise_image_tensor, mask_expand2)
                mask_noise_pil = Image.fromarray((noise_masked_tensor.cpu().numpy()).astype(np.uint8)).convert("RGBA")
        return mask_noise_pil

    def __call__(self, image):
        img = np.asarray(image)
        prev = Image.fromarray(img).convert("RGBA")
        res = prev
        if random.random() < self.p1:
            mask_noise_pil = self.get_segment_noise(img, self.alpha)
            if not (mask_noise_pil is None):
                res = Image.alpha_composite(prev, mask_noise_pil)
        res = res.convert('RGB')
        return res

class Randboxpost(object):
    def __init__(self, p1, alpha):
        self.p1 = p1
        self.alpha = alpha

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.zeros((h, w), np.float32)
        res = img
        if random.random() < self.p1:
            y = np.random.randint(h)
            x = np.random.randint(w)
            length = np.random.randint(int(np.sqrt(1 * h)), h)
            width  = np.random.randint(int(np.sqrt(1 * w)),w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - width // 2, 0, w)
            x2 = np.clip(x + width // 2, 0, w)

            mask[y1: y2, x1: x2] = 1.
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            box = img * mask
            noise = torch.zeros([3,h,w])
            flag = False
            for c in range(3):
                tmp_channel = box[c, :, :]
                # x = tmp_channel[tmp_channel != 0]
                x = tmp_channel
                if len(x) > 0:
                    x = x.double()
                    std, mean = torch.std_mean(x)
                    try:
                        normal_beta = torch.distributions.Normal(loc=mean, scale=std)
                        beta = normal_beta.sample([h * w]).reshape(h, w)
                        noise[c, :, :] = beta
                        flag = True
                    except:
                        break
            if flag:
                mask_bar = 1 - mask
                res = res * mask_bar + (1 - self.alpha / 255.0) * mask * res + self.alpha / 255.0 * mask * noise
                res = torch.clamp(res, min=0., max=255.)
        return res         

class Randpepperpost(object):
    def __init__(self, sl=0.02, sh=0.4, alpha=1):
        self.sl = sl
        self.sh = sh
        self.alpha = alpha

    def generate_pepper_mask(self, image):
        _, h, w = image.shape  # cifar-100 is convert to HWC by torchvision
        mask = torch.zeros(h, w)
        p1 = random.uniform(self.sl, self.sh)
        p = int(p1 * h * w)  # num_ones
        flat_mask = mask.view(-1)
        indices = torch.randperm(flat_mask.numel())[:p]
        flat_mask[indices] = 1.
        mask = flat_mask.view(h, w)
        return mask

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = self.generate_pepper_mask(img)
        res = img
        std_scale = 1.0  # dist.sample() # 4.0
        omn, omx = torch.min(img), torch.max(img)
        mask = mask.expand_as(img)
        box = img * mask
        noise = torch.zeros([3, h, w])
        flag = False
        for c in range(3):
            tmp_channel = img[c, :, :]  # box[c, :, :]
            x = tmp_channel  # [tmp_channel != 0]
            if len(x) > 0:
                x = x.double()
                std, mean = torch.std_mean(x)
                try:
                    normal_beta = torch.distributions.Normal(loc=mean, scale=std * std_scale)
                    beta = normal_beta.sample([h * w]).reshape(h, w)
                    noise[c, :, :] = beta
                    flag = True
                except:
                    break
        if flag:
            mask_bar = 1 - mask
            res = res * mask_bar + (1 - self.alpha / 255.0) * mask * res + self.alpha / 255.0 * mask * noise
            res = torch.clamp(res, min=omn, max=omx)
        return res

class Randimgpost(object):
    def __init__(self, p1, alpha):
        self.p1 = p1
        self.alpha = alpha

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        res = img
        std_scale = 1.0  # dist.sample() # 1.0
        omn, omx = torch.min(img), torch.max(img)
        if random.random() < self.p1:
            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            # box = img * mask
            noise = torch.zeros([3, h, w])
            flag = False
            for c in range(3):
                tmp_channel = img[c, :, :]  # box[c, :, :]
                x = tmp_channel  # [tmp_channel != 0]
                if len(x) > 0:
                    x = x.double()
                    std, mean = torch.std_mean(x)
                    try:
                        normal_beta = torch.distributions.Normal(loc=mean, scale=std * std_scale)
                        beta = normal_beta.sample([h * w]).reshape(h, w)
                        noise[c, :, :] = beta
                        flag = True
                    except:
                        break
            if flag:
                mask_bar = 1 - mask
                res = res * mask_bar + (1 - self.alpha / 255.0) * mask * res + self.alpha / 255.0 * mask * noise
                res = torch.clamp(res, min=omn, max=omx)
        return res