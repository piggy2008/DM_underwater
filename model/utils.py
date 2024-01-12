import torch
from model.ddpm_trans_modules.unet import UNet

categories = ['fish', 'marine life', 'coral', 'rock', 'diving', 'deep see',
                      'wreckage', 'sculpture', 'caves', 'underwater stuff']

def load_part_of_model(new_model, src_model_path, s):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        if k in m_dict.keys():
            param = src_model.get(k)
            if param.shape == m_dict[k].data.shape:
                m_dict[k].data = param
                print('loading:', k)
            else:
                print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict, strict=s)
    return new_model

def load_part_of_model2(new_model, src_model_path):
    src_model = torch.load(src_model_path, map_location='cuda:1')
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        # print('param key:', k)
        k2 = k.replace('denoise_fn.', '')
        if k2 in m_dict.keys():
            # print(k)
            param = src_model.get(k)
            if param.shape == m_dict[k2].data.shape:
                m_dict[k2].data = param
                print('loading:', k)
            # else:
            #     print('shape is different, not loading:', k)
        else:
            print('not loading:', k)

    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model3(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    for k in src_model.keys():
        param = src_model.get(k)
        if (k.find('style_loss') > -1):
            continue
        elif (k.find('encoder_water.conv2') > -1):
            print('loading control:', k)
            k2 = k.replace('encoder_water.conv2', 'encoder_water.conv2_control')
            m_dict[k2].data = param
        elif(k.find('encoder_water.conv3') > -1):
            print('loading control:', k)
            k2 = k.replace('encoder_water.conv3', 'encoder_water.conv3_control')
            m_dict[k2].data = param
        elif (k.find('encoder_water.conv4') > -1):
            print('loading control:', k)
            k2 = k.replace('encoder_water.conv4', 'encoder_water.conv4_control')
            m_dict[k2].data = param
        elif (k.find('encoder_water.block1') > -1):
            print('loading control:', k)
            k2 = k.replace('encoder_water.block1', 'encoder_water.block1_control')
            m_dict[k2].data = param
        elif (k.find('encoder_water.block2') > -1):
            print('loading control:', k)
            k2 = k.replace('encoder_water.block2', 'encoder_water.block2_control')
            m_dict[k2].data = param
        elif (k.find('encoder_water.block3') > -1):
            print('loading control:', k)
            k2 = k.replace('encoder_water.block3', 'encoder_water.block3_control')
            m_dict[k2].data = param
        elif (k.find('encoder_water.block4') > -1):
            print('loading control:', k)
            k2 = k.replace('encoder_water.block4', 'encoder_water.block4_control')
            m_dict[k2].data = param
        elif (k.find('encoder_water.conv1') > -1):
            print('loading input channel:', k)
            print(param.shape)
            param_temp = torch.zeros_like(m_dict[k].data)
            param_temp[:, 0, :, :] = param[:, 0, :, :]
            param_temp[:, 1, :, :] = param[:, 0, :, :]
            param_temp[:, 2:, :, :] = param
            m_dict[k].data = param_temp
        else:
            if param.shape == m_dict[k].data.shape:
                m_dict[k].data = param
                print('loading:', k)
            else:
                print('shape is different, not loading:', k)
    new_model.load_state_dict(m_dict)
    return new_model

def load_part_of_model4(new_model, src_model_path):
    src_model = torch.load(src_model_path)
    m_dict = new_model.state_dict()
    new_model_keys = list(m_dict.keys())
    for k in src_model.keys():
        param = src_model.get(k)
        if (k.find('style_loss') > -1):
            continue
        elif (k.find('x_embedder.proj.weight') > -1):
            print('loading input channel:', k)
            print(param.shape)
            param_temp = torch.zeros_like(m_dict[k].data)
            param_temp[:, 0, :, :] = param[:, 0, :, :]
            param_temp[:, 1, :, :] = param[:, 0, :, :]
            param_temp[:, 2:, :, :] = param
            m_dict[k].data = param_temp
        else:
            if param.shape == m_dict[k].data.shape:
                m_dict[k].data = param
                print('loading:', k)
                new_model_keys.remove(k)
            else:
                print('shape is different, not loading:', k)
    for k in new_model_keys:
        if (k.find('.blocks2.') > -1):
            k2 = k.replace('.blocks2.', '.blocks.')
            param2 = src_model.get(k2)
            if param2.shape == m_dict[k].data.shape:
                m_dict[k].data = param2
                print('loading extra layer:', k)
        elif (k.find('.transformer_blocks2.') > -1):
            k2 = k.replace('.transformer_blocks2.', '.transformer_blocks.')
            param2 = src_model.get(k2)
            if param2.shape == m_dict[k].data.shape:
                m_dict[k].data = param2
                print('loading extra layer:', k)
    new_model.load_state_dict(m_dict)
    return new_model







