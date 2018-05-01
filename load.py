
def load_state_dict(model, path):
    state_dict = torch.load('my_file.pt', map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    return model
