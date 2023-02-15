import pathlib
import time

import numpy as np

from utils.extract_frames import Extractor,Frame,Segmentor
from utils.inference_utils import Segmentor,pil_to_cv,cv_to_pil
from utils.tagger import Tagger
import cv2
from detectron2.structures.instances import Instances
from dataset import  *
# video = "Lyco.mp4"
# output_dir = "extracted"
# # e = Extractor(video,output_dir)
# # e.extract_keyframes(0.3)
# # e.collect_frames()
# # e.extract_clips(1,300)
#
# config_file = r"D:\pycharmWorkspace\flaskProj\utils\configs\CondInst\CondInst-AnimeSeg.yaml"
# model_file = r"D:\pycharmWorkspace\flaskProj\utils\models\CondInst-AnimeSeg.pth"
# s = Segmentor(config_file,model_file)



# model = YOLO("yolov8s.pt")
# imgs,boxes_list = dir_to_imgboxes(model=model,dir_path="temp")
# make_dataset(imgs,boxes_list,output_dir="testout")

'''
Issue:
MAD:frequently switches scenes back and forth



'''
# e = Extractor(None,output_dir="temp")
# e.video = "MAD_mid.mp4"
#
# e.frames = e.extract_keyframes(0.1)
# frames =  e.extract_refs(e.frames)

'''
TODO:
1. remove duplicate ref imgs
2. improve algorithm to search for characters

3 qol parts for data:
1. filter out tags on record (kinda easy)
2. find close tags/typo(auto suggestion) (difficult)
3. tags random sampling (easy)


'''

# from sentence_transformers import SentenceTransformer,util
# from PIL import Image
# print("Loading CLIP model...")
# model = SentenceTransformer('clip-ViT-B-32')
# # Next we compute the embeddings
# # To encode an image, you can use the following code:
# # from PIL import Image
# img1 = "similar/104 0.8834910988807678.jpg"
# imgs = ["similar/104 0.8834910988807678.jpg",
#         "similar/106 0.8422887921333313.jpg",
#         "sakuga_images/53.jpg"]
#
# encoded_images = model.encode([Image.open(img) for img in imgs])
#
# processed = util.paraphrase_mining_embeddings(encoded_images)
# print(processed)

# print(encoded_image.shape)
# print(type(encoded_image))


# image_names = list(glob.glob('./*.jpg'))
# print("Images:", len(image_names))
# encoded_image = model.encode([Image.open(filepath) for filepath in image_names], batch_size=128, convert_to_tensor=True, show_progress_bar=True)
# print(encoded_image.shape)
# print(encoded_image)

# Now we run the clustering algorithm. This function compares images aganist
# all other images and returns a list with the pairs that have the highest
# cosine similarity score
# processed_images = util.paraphrase_mining_embeddings(encoded_image)
# NUM_SIMILAR_IMAGES = 10

# e.extract_refs_onestage(video)

# img = cv2.imread("28.jpg")
# # cv2.imshow('Original',img)
# output:Instances
# output = s(img)['instances']
# fields = output.get_fields()
# print(fields.keys())
# print(fields['pred_boxes'])

# boxes = s.get_boxes(fields)
# img_box = s.draw_boxes(img,boxes)
# cv2.imshow("predicted",img_box)
# cropped_imgs = s.crop_boxes(img,boxes)
# for i in range(len(cropped_imgs)):
#     print(cropped_imgs[i].shape)
#     cv2.imshow(f'Image {i}',cropped_imgs[i])

# masks = s.get_masks(fields)

# masked = s.mask_img(img,masks)
# i = 0
# for mask in masked:
#     print(mask[0],type(mask[1]), mask[1].shape)
#     cv2.imshow(f"{mask[0]}",mask[1])

# merged = s.merge_masks([x[1] for x in masks] )
# cv2.imshow("Merged",merged)
#
# cv2.waitKey(-1)

# pred_boxes = fields['pred_boxes']
# for box in pred_boxes:
#     pt1 = (int(box[0]),int(box[1]))
#     pt2 = (int(box[2]),int(box[3]))
#     cv2.rectangle(img,pt1,pt2,(255,0,0),2)
# cv2.imshow("Pred boxes", img)
# cv2.waitKey(-1)

# img1 = cv2.imread("15.jpg")
# img2 = cv2.imread("32.jpg")
# img1 = cv_to_pil(img1)
# img2 = cv_to_pil(img2)
# t = Tagger()
# d1 = t(img1)
# d2 = t(img2)
# t1 = t.dict_to_tuples(d1)
# t2 = t.dict_to_tuples(d2)
# print(t1)
# print(t2)
# print(t.cos_sim(d1,d2))

# frames_dir = "sakuga_images"
# p = pathlib.Path(frames_dir)
# frames = list(p.iterdir())
#
# imgs = []
# for f in frames:
#     img = cv2.imread(str(f.resolve()))
#
#     shape = [int(x) for x in img.shape]
#     img = cv2.resize(img,(shape[1],shape[0]),cv2.INTER_CUBIC )
#     # print(img.shape)
#     # cv2.imshow("resized",img)
#     # cv2.waitKey(-1)
#     # exit(0)

    # img = cv2.resize(img,)
    # imgs.append(img)

# avg = []
# for i in range(4):
#     start = time.time()
#     for idx,img in enumerate(imgs):
#         s(img)
#         print(f"done {idx}")
#     end = time.time()
#     avg.append(float(end-start))
# avg = np.asarray(avg).mean()
# print(f"Inference for {len(frames)} took {avg}s on average!")
'''
/2 took a little faster


non-resized took 30-40s
conclusion: doesn't help to resize. 
'''

