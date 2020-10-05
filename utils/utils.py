from models.basenet import *
import torch


def get_model_mme(net, num_class=13, unit_size=2048, temp=0.05):
    model_g = ResBase(net, unit_size=unit_size)
    model_c = ResClassifier_MME(num_classes=num_class, input_size=unit_size, temp=temp)
    return model_g, model_c


def save_model(model_g, model_c, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c_state_dict': model_c.state_dict(),
    }
    torch.save(save_dic, save_path)


def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c

