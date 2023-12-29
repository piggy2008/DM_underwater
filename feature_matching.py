import numpy as np

feature_underwater = np.load('feature_underwater.npy', allow_pickle=True)
feature_imagenet = np.load('feature_imagenet.npy', allow_pickle=True)
match_image = {}
for name_u, feature_u in feature_underwater.item().items():
    # print(name_u, '----', feature_u[0].shape)
    min_dis = 1000000
    match_name = ''
    for name_i, feature_i in feature_imagenet.item().items():
        cur_dis = np.linalg.norm(feature_u[0] - feature_i[0])
        if cur_dis < min_dis:
            min_dis = cur_dis
            match_name = name_i

    match_image[name_u] = match_name
    print(name_u, '----', match_name)

print(match_image)
np.save('match_image.npy', match_image)