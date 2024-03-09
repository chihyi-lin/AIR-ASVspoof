import pickle
from librosa.util import find_files
import scipy.io as sio
import sys
import os

# script is used to reload LFCC (Linear Frequency Cepstral Coefficients) data from .mat files, 
# and save them as pickle (.pkl) files

access_type = "LA"
# # on air station gpu
path_to_ASVspoof19_mat = '/Users/chihyi/Documents/CL_WS_24/Speech_tech/project/AIR-ASVspoof/Features'
path_to_ASVspoof19_features = '/Users/chihyi/Documents/CL_WS_24/Speech_tech/project/AIR-ASVspoof/ASVspoof2019_'+access_type+'_Features/'
path_to_in_the_wild_mat = '/Users/chihyi/Documents/CL_WS_24/Speech_tech/project/AIR-ASVspoof/in_the_wild_MidFeatures'
path_to_in_the_wild_features = '/Users/chihyi/Documents/CL_WS_24/Speech_tech/project/AIR-ASVspoof/in_the_wild_Features'

def reload_data(path_to_features, part):
        matfiles = find_files(path_to_ASVspoof19_mat + '/' + part + '/', ext='mat')
        for i in range(len(matfiles)):
            if matfiles[i][len(path_to_ASVspoof19_mat)+len(part)+2:].startswith('LFCC'):
                key = matfiles[i][len(path_to_ASVspoof19_mat) + len(part) + 7:-4]
                lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
                with open(path_to_features + part +'/'+ key + 'LFCC.pkl', 'wb') as handle2:
                    pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)

def reload_in_the_wild():
     matfiles = find_files(path_to_in_the_wild_mat + '/', ext='mat')
     for i in range(len(matfiles)):
          file_name = matfiles[i][len(path_to_in_the_wild_mat)+1:]  # file_name = "x.mat"
          if file_name.endswith('.mat'):
               key = os.path.splitext(file_name)[0]
               lfcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
               with open(path_to_in_the_wild_features + '/' + key + 'LFCC.pkl', 'wb') as handle2:
                    pickle.dump(lfcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)
               
     

if __name__ == "__main__":
    dataset = sys.argv[1]   # "ASVspoof2019" or "in_the_wild"

    if dataset == "ASVspoof2019":
        reload_data(path_to_ASVspoof19_features, 'train')
        reload_data(path_to_ASVspoof19_features, 'dev')
        reload_data(path_to_ASVspoof19_features, 'eval')

    elif dataset == "in_the_wild":
         reload_in_the_wild()