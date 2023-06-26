import torch.utils.data as data
import numpy as np
import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat

class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.is_normal = is_normal
        self.dataset = args.dataset
        
        if self.dataset == 'shanghai':
            self.seg_len = 112
            if test_mode:
                self.rgb_list_file = 'list/shanghai-i3d-test.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train.list'
        else:
            self.seg_len = 405
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d-train.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            features = features.transpose(1, 0, 2)  
            divided_features = []
            divided_masks = []
            for feature in features:
                feature, mask = self.pad_feat(feature)
                divided_features.append(feature)
                divided_masks.append(mask)
            divided_features = np.array(divided_features, dtype=np.float32)
            divided_masks = np.array(divided_masks)
            return divided_features, label, divided_masks
    def pad_feat(self, feature):
        mask = np.ones((1, self.seg_len))
        if self.seg_len < feature.shape[0]:
            feature = process_feat(feature, self.seg_len) 
            return feature, mask
        else:
            t = self.seg_len - feature.shape[0]
            mask[0, feature.shape[0] : self.seg_len] = 0
            return np.pad(feature, ((0, t), (0, 0))), mask
        
    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
