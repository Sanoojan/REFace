class CelebAHQDataset(Dataset):
    def __init__(self, dataset_root, mode="test",
                 img_transform=TO_TENSOR, label_transform=TO_TENSOR,
                 load_vis_img=False,fraction=1.0,
                 flip_p=-1):  # negative number for no flipping

        self.mode = mode
        self.root = dataset_root
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.load_vis_img = load_vis_img
        self.fraction=fraction
        self.flip_p = flip_p

        if mode == "train":
            self.imgs = sorted([osp.join(self.root, "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000)])
            self.labels =  sorted([osp.join(self.root, "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000)]) 
            self.labels_vis =  sorted([osp.join(self.root, "vis", "%d.png"%idx) for idx in range(28000)]) if self.load_vis_img else None
        else:
            self.imgs = sorted([osp.join(self.root, "CelebA-HQ-img", "%d.jpg"%idx) for idx in range(28000, 30000)])
            # self.labels = ([osp.join(self.root, "CelebA-HQ-mask", "%d"%int(idx/2000) ,'{0:0=5d}'.format(idx)+'_skin.png') for idx in range(28000, 30000)])
            self.labels =  sorted([osp.join(self.root, "CelebA-HQ-mask/Overall_mask", "%d.png"%idx) for idx in range(28000, 30000)]) 
            self.labels_vis =  sorted([osp.join(self.root, "vis", "%d.png"%idx) for idx in range(28000, 30000)]) if self.load_vis_img else None

        self.imgs= self.imgs[:int(len(self.imgs)*self.fraction)]
        self.labels= self.labels[:int(len(self.labels)*self.fraction)]
        self.labels_vis= self.labels_vis[:int(len(self.labels_vis)*self.fraction)]  if self.load_vis_img else None

        if self.load_vis_img:
            assert len(self.imgs) == len(self.labels) == len(self.labels_vis)
        else:
            assert len(self.imgs) == len(self.labels)

        # image pairs indices
        self.indices = np.arange(len(self.imgs))


    def __len__(self):
        return len(self.indices)

    def load_single_image(self, index):
        """Load one sample for training, inlcuding 
            - the image, 
            - the semantic image, 
            - the corresponding visualization image

        Args:
            index (int): index of the sample
        Return:
            img: RGB image
            label: seg mask
            label_vis: visualization of the seg mask
        """
        img = self.imgs[index]
        img = Image.open(img).convert('RGB')
        if self.img_transform is not None:
            img = self.img_transform(img)

        label = self.labels[index]
        # print(label)
        label = Image.open(label).convert('L')
        # breakpoint()
        # label2=TO_TENSOR(label)
        # save_image(label2, str(index)+'_label.png')
        # save_image(img, str(index)+'_img.png')  
        
        if self.label_transform is not None:
            label= self.label_transform(label)
 

        if self.load_vis_img:
            label_vis = self.labels_vis[index]
            label_vis = Image.open(label_vis).convert('RGB')
            label_vis = TO_TENSOR(label_vis)
        else:
            label_vis = -1  # unified interface
        # save_image(label, str(index)+'_label.png')
        # save_image(img, str(index)+'_img.png')  

        return img, label, label_vis

    def __getitem__(self, idx):
        index = self.indices[idx]
        img, label, label_vis = self.load_single_image(index)
        if self.flip_p > 0:
            if random.random() < self.flip_p:
                img = TF.hflip(img)
                label = TF.hflip(label)
           
        return img, label, label_vis
    