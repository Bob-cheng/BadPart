import torch


def import_optical_flow_model(model_pth, args):
    assert model_pth is not None,'path of model weights is none'
    if args.model_name == 'FlowNetC':
        from FlowNetworks.flow_models.models.FlowNetC import FlowNetC
        model = FlowNetC()
        weights = torch.load(model_pth)
        model.load_state_dict(weights['state_dict'])
    elif args.model_name == 'FlowNet2':
        from FlowNetworks.flow_models.models.flownet2_models import FlowNet2
        model = FlowNet2()
        weights = torch.load(model_pth)
        model.load_state_dict(weights['state_dict'])
    elif args.model_name == 'PWC-Net':
        from FlowNetworks.flow_models.models.PWCNet import PWCDCNet
        model = PWCDCNet()
        weights = torch.load(model_pth)
        model.load_state_dict(weights)
    else:
        raise RuntimeError('model does not supported')

    return model

