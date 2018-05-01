def load_state_dict(model, path):
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    return model
