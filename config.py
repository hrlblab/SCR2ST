

#commented out variables are handled by argparse in main.py
debug = True

lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
# epochs = 10

model_name = 'resnet50'
image_embedding = 2048
spot_embedding = 300 #number of shared hvgs (change for each dataset)

pretrained = True
trainable = True 
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1


#  'vit_base_patch32_224',
#  'vit_base_patch32_224_clip_laion2b',
#  'vit_base_patch32_224_in21k',
#  'vit_base_patch32_224_sam',
