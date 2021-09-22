import pickle
import numpy as np
import cupy as cp

class volume():
    def __init__(self, data_dict):
        """
        """
        self.name = data_dict['name']
        self.dtype = data_dict['dtype']
        self.projection_path = data_dict['projection path']
        self.data = data_dict
        self.volume = self.read_volume()
    
    def read_field(self, files):
        pass

    def read_volume(self, mode):
        if mode == 'binary':
            volume = pickle.load(open(self.data['binary path'],'rb'))
        elif mode == 'raw':
            read_fcn = data_dict['imread function']
            projection_files = glob(data_dict['projection path'])
            n_proj = len(projection_files)
            height,width = np.asarray(read_fcn(projection_files[0])).shape
            volume = np.zeros([n_proj,height,width])
            for i in tqdm(range(n_proj)):
                volume[i,:,:]  = np.asarray(read_fcn(proj_files[i]), dtype = self.dtype)
        return volume

    def dump_volume(self, dump_path):
        pickle.dump(self.volume,open(dump_path,'wb'))

    def __call__(self):
        return self.volume
