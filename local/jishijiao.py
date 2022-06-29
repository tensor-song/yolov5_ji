def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    # sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    
    # return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
    return [os.path.join('/project/train/src_repo/dataset/labels', x.rsplit(os.sep, 2)[-2],x.rsplit(os.sep, 2)[-1].split('.')[0]+'.txt') for x in img_paths]

