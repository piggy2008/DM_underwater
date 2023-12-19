import torch
from torch import nn
import torchvision.transforms as transforms
from transformers import CLIPTokenizer, CLIPTextModel
# from model.spatial_attention import SpatialTransformer
import clip



def generate_src_target_txt(labels):
    src_txt_list = []
    target_txt_list = []
    for i in range(labels.shape[0]):
        label = labels[i].data.cpu().numpy()
        if label == 0:
            src_txt = 'fish in the underwater'
            target_txt = 'colorful fish and the contrast is strong between the foreground and background'
        elif label == 1:
            src_txt = 'marine life'
            target_txt = 'colorful marine life and the contrast is strong between the foreground and background'
        elif label == 2:
            src_txt = 'coral in the underwater'
            target_txt = 'colorful coral on land and the contrast is strong between the foreground and background'
        elif label == 3:
            src_txt = 'rock in the underwater'
            target_txt = 'rock on land and the contrast is strong between the foreground and background'
        elif label == 4:
            src_txt = 'diving people in the underwater'
            target_txt = 'diving people on land and the contrast is strong between the foreground and background'
        elif label == 5:
            src_txt = 'deep see scenes'
            target_txt = 'scenes without underwater condition and the contrast is strong between the foreground and background'
        elif label == 6:
            src_txt = 'wreckage in the underwater'
            target_txt = 'clean wreckage and the contrast is strong between the foreground and background'
        elif label == 7:
            src_txt = 'sculpture in the underwater'
            target_txt = 'sculpture on land and the contrast is strong between the foreground and background'
        elif label == 8:
            src_txt = 'caves in the underwater'
            target_txt = 'normal caves and the contrast is strong between the foreground and background'
        else:
            src_txt = 'underwater stuff'
            target_txt = 'stuff in the air and the contrast is strong between the foreground and background'
        src_txt_list.append(src_txt)
        target_txt_list.append(target_txt)
    return src_txt_list, target_txt_list

class CLIP_loss(torch.nn.Module):
    def __init__(self, device=0, model_name='ViT-B/32'):
        super(CLIP_loss, self).__init__()
        self.device = device
        model, preprocess = clip.load(model_name, device=device)
        self.model = model
        self.model.eval().requires_grad_(False)
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                                                                                 2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                             preprocess.transforms[4:])
        self.L1_loss = torch.nn.L1Loss()
        self.cosine_loss = torch.nn.CosineSimilarity()

    def forward(self, src_img, target_img, labels):
        src_img = self.preprocess(src_img)
        target_img = self.preprocess(target_img)
        src_img_features = self.model.encode_image(src_img)
        target_img_features = self.model.encode_image(target_img)
        src_img_features /= src_img_features.clone().norm(dim=-1, keepdim=True)
        target_img_features /= target_img_features.clone().norm(dim=-1, keepdim=True)

        edit_direction_img = target_img_features - src_img_features
        edit_direction_img /= (edit_direction_img.clone().norm(dim=-1, keepdim=True) + 1e-7)
        # print(src_img_features.shape)
        src_txt_inputs, target_txt_inputs = generate_src_target_txt(labels)
        src_txt_features = self.model.encode_text(torch.cat(src_txt_inputs).to(self.device))
        target_txt_features = self.model.encode_text(torch.cat(target_txt_inputs).to(self.device))
        src_txt_features /= src_txt_features.clone().norm(dim=-1, keepdim=True)
        target_txt_features /= target_txt_features.clone().norm(dim=-1, keepdim=True)
        # print(src_txt_features.shape)

        text_direction = (target_txt_features - src_txt_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        # dis = 1. - self.cosine_loss(torch.cat([src_img_features, src_txt_features], dim=-1),
        #                             torch.cat([target_img_features, target_txt_features], dim=-1))
        dis = 1. - self.cosine_loss(edit_direction_img, text_direction)
        dis = dis.mean()
        dis = -torch.log((2 - dis) / 2)
        # src_fuse_features = torch.cat([src_img_features, src_txt_features], dim=-1)
        # target_fuse_features = torch.cat([target_img_features, target_txt_inputs], dim=-1)
        return dis

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


if __name__ == '__main__':
    # clip_loss = CLIP_loss()
    # path = '/home/ty/code/DM_underwater/dataset/water_train_16_128/hr_128_real/00001.png'
    # from torchvision import transforms
    # # image = Image.open(path).convert("RGB")
    # # image = transforms.ToTensor()(image).unsqueeze(0).to('cuda:0')
    # image = torch.randn([2, 3, 128, 128]).to('cuda:0')
    # image.clip_(-1, 1)
    # labels = torch.tensor([[1], [2]]).to('cuda:0')
    # clip_loss(image, image, labels)
    clip = FrozenCLIPEmbedder().to('cuda:0')
    a = ['underwater', 'air']
    output = clip.encode(a)

    print(output.shape)
    # trans = SpatialTransformer(320, 1, 320, context_dim=768).to('cuda:0')
    # input = torch.randn([1, 320, 64, 64]).to('cuda:0')
    # a = trans(input, output)
    # print(a.shape)




