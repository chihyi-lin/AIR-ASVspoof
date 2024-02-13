import pickle
from librosa.util import find_files
import scipy.io as sio

# script is used to reload LFCC (Linear Frequency Cepstral Coefficients) data from .mat files, 
# and save them as pickle (.pkl) files

access_type = "LA"
# # on air station gpu
path_to_mat = '/Users/chihyi/Documents/CL_WS_24/Speech_tech/project/AIR-ASVspoof/Features'
path_to_audio = '/data/neil/DS_10283_3336/'+access_type+'/ASVspoof2019_'+access_type+'_'
path_to_features = '/Users/chihyi/Documents/CL_WS_24/Speech_tech/project/AIR-ASVspoof/ASVspoof2019_'+access_type+'_Features/'

def reload_data(path_to_features, part):
    matfiles = find_files(path_to_mat + '/' + part + '/', ext='mat')
    for i in range(len(matfiles)):
        if matfiles[i][len(path_to_mat)+len(part)+2:].startswith('LFCC'):
            key = matfiles[i][len(path_to_mat) + len(part) + 7:-4]
            lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
            with open(path_to_features + part +'/'+ key + 'LFCC.pkl', 'wb') as handle2:
                pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    reload_data(path_to_features, 'train')
    reload_data(path_to_features, 'dev')
    reload_data(path_to_features, 'eval')
