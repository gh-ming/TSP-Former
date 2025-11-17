# import pickle
# file_path = '/nfs/project/netdisk/100/data/global_datasets/Global-Scale/train/region_0_graph_gt.pickle'
# file = open(file_path,'rb')
# data = pickle.load(file)
# print(type(data))
# for i,(k,v) in enumerate(data.items()):
#     if i in range(0,10):
#         print(k,v)
import tarfile
import tqdm
tar_path = '/nfs/project/netdisk/100/data/global_datasets/BigEarthNet/BigEarthNet-S2.tar/BigEarthNet-S2.tar'
with tarfile.open(tar_path) as tar:
    members = tar.getmembers()
    with open('tar_contents.txt','w') as f:
        for member in members:
            # if member.isdir():
            path = member.name.replace('\\','/')
            f.write(path +'\n')
            
    

