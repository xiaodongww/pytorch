class labelConverter():
    def __init__(self, meta_path="/home/hechen/ILSVRC/ILSVRC2012_devkit_t12/data/meta.mat"):
        import scipy.io as sio
        synsets  = sio.loadmat(meta_path)['synsets']
        height_count = {i:[] for i in range(20)}
        new_synsets = {}
        wnid_to_ilsid = {}
        for synset in synsets:
            ils_id = synset[0][0][0][0]
            wordnet_height = synset[0][-2][0][0]
            children = synset[0][-3][0]
            num_children = len(children)
            words = synset[0][2][0]
            wnid = synset[0][1][0]
            height_count[wordnet_height].append(ils_id)
            new_synsets[ils_id] = {}
            new_synsets[ils_id]['wnid'] = wnid
            new_synsets[ils_id]['words'] = words
            new_synsets[ils_id]['wordnet_height'] = wordnet_height
            new_synsets[ils_id]['num_children'] = num_children
            new_synsets[ils_id]['children'] = children
            wnid_to_ilsid[wnid] = ils_id
        self.new_synsets = new_synsets
    def label_to_word(self, label):
        return self.new_synsets[label]['words']
converter = labelConverter()
print(converter.label_to_word(13))
