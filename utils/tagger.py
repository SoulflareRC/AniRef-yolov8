import tempfile

import PIL.Image
from deepdanbooru_onnx import DeepDanbooru,process_image
import math
import numpy as np
import cv2
from PIL import  Image
from collections import Counter
import gdown
from pathlib import Path
import os,shutil
class Tagger(object):
    '''
    Wrapper class for deepdanbooru utility
    '''
    def __init__(self):
        self.d = DeepDanbooru(threshold=0.5)
        self.threshold_s = 0.4
        self.toi = self.grab_toi()
        self.chara_tags = {
        #     in the form of name: tags
        }
    def __call__(self,img:Image.Image):
        '''
        Wrapped call of deepdanbooru tagger
        :param img:
        :return: a dict
        '''
        img = process_image(img)
        return self.d(img)
    def grab_toi(self):
        toi = set() #tag of interest
        tag_desc_txt = Path("tags_desc.txt")
        tag_desc_link = "https://drive.google.com/uc?id=1e4iSJCrtyFaeipYQqx0feVQnI4z7H9LQ"
        tag_desc_link = "https://drive.google.com/file/d/1e4iSJCrtyFaeipYQqx0feVQnI4z7H9LQ/view?usp=share_link"
        tag_chara_txt = Path("tags_character.txt")
        tag_chara_link = "https://drive.google.com/uc?id=G2hw8HTzRSmzD77SxZ3pxznEeG0ACRRA"
        tag_chara_link = "https://drive.google.com/file/d/1G2hw8HTzRSmzD77SxZ3pxznEeG0ACRRA/view?usp=share_link"
        if not tag_desc_txt.exists():
            gdown.download(url=tag_desc_link,output=tag_desc_txt.name,fuzzy=True)
        if not tag_chara_txt.exists():
            gdown.download(url=tag_chara_link,output=tag_chara_txt.name,fuzzy=True)
        with open(tag_desc_txt,"r") as f:
            tags = f.readlines()
            for tag in tags:
                tag = tag.replace(" ","_").replace("\n","")
                toi.add(tag)
        with open(tag_chara_txt,"r") as f:
            tags = f.readlines()
            for tag in tags:
                tag = tag.replace(" ","_").replace("\n","")
                toi.add(tag)
        print(f"Grabbed {len(toi)} tags of interest")
        return toi
    def tag_chara(self,img):
        '''
        :param img: will be a cv2 image for compatibility, but needs to be converted into PIL image since deepdanbooru works with PIL Images
        :return: predicted tags from toi
        '''
        if type(img)==np.ndarray:
            # gradio seems to be doing some internal convertion that converts the image to RGB first
            # print(img.mode)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            # img.show("what")
        d = self(img)
        tags = set(d.keys()).intersection(self.toi)  # extract toi
        return tags
    def comp_tags(self,tags1,tags2):
        '''
        :param tags1: tags of the first chara
        :param tags2: tags of the second chara
        :return: similarity of two lists
        similarity is calculated by taking the shorter tags and see how much of the intersection does it have.
        '''
        if len(tags1) < len(tags2):
            short = tags1
            long = tags2
        else:
            short = tags2
            long = tags1
        intersection = short.intersection(long)
        print("intersection:",intersection)
        similarity = 0 if len(short) < 4 else len(intersection) / len(short)
        return similarity

    def mark_chara_from_folder(self,folder_path,charas:list[str],similarity_threshold = 0.4):
        '''
        Identify which character is in this image
        :param img: a cv2 images
        :param charas:names of characters to identify from
        :return: a single character with the highest similarity and similarity
        '''
        # if len(charas)<1:
        #     return None,0.0
        # else:
        #     tags =self.tag_chara(img)
        #     best_match = charas[0]
        #     best_match_sim = 0
        #     for chara in charas:
        #         config = self.charas[chara]
        #         chara_tags = set(config['tags'])
        #         similarity = self.comp_tags(chara_tags,tags)
        #         if similarity>best_match_sim:
        #             best_match=chara
        #             best_match_sim=similarity
        #     return best_match,best_match_sim
        folder = Path(folder_path)
        res_folders = {
            # name:folder path
        }
        fs = list(folder.iterdir())
        for f in fs:
            if not f.is_dir():
                img = Image.open(f)
                tags = self.tag_chara(img)
                if type(tags) is not set:
                    tags = set(tags)
                best_match_chara = "None"
                best_match_sim   = 0
                for chara in charas:
                    chara_tags = self.chara_tags[chara]
                    if type(chara_tags) is not set:
                        chara_tags = set(chara_tags)
                    similarity = self.comp_tags(tags,chara_tags)
                    if similarity>similarity_threshold and similarity>best_match_sim:
                        best_match_chara = chara
                target_folder = folder.joinpath(best_match_chara)
                res_folders[best_match_chara]=target_folder
                if not target_folder.exists():
                    os.makedirs(target_folder)
                target_f = target_folder.joinpath(f.name)
                shutil.copy(f,target_f)
        return res_folders
    def mark_chara_from_imgs(self,imgs:list[Image],charas:list[str],output_folder,similarity_threshold = 0.4):
        '''
        Identify which character is in this image
        :param img: a cv2 images
        :param charas:names of characters to identify from
        :return: a single character with the highest similarity and similarity
        '''
        # if len(charas)<1:
        #     return None,0.0
        # else:
        #     tags =self.tag_chara(img)
        #     best_match = charas[0]
        #     best_match_sim = 0
        #     for chara in charas:
        #         config = self.charas[chara]
        #         chara_tags = set(config['tags'])
        #         similarity = self.comp_tags(chara_tags,tags)
        #         if similarity>best_match_sim:
        #             best_match=chara
        #             best_match_sim=similarity
        #     return best_match,best_match_sim
        output_folder = Path(output_folder)
        if not output_folder.exists():
            os.makedirs(output_folder)
        res_folders = {
            # name:folder path
        }
        for img in imgs:
                img:Image
                tags = self.tag_chara(img)
                if type(tags) is not set:
                    tags = set(tags)
                best_match_chara = "None"
                best_match_sim   = 0
                for chara in charas:
                    chara_tags = self.chara_tags[chara]
                    if type(chara_tags) is not set:
                        chara_tags = set(chara_tags)
                    similarity = self.comp_tags(tags,chara_tags)
                    if similarity>similarity_threshold and similarity>best_match_sim:
                        best_match_chara = chara
                target_folder = output_folder.joinpath(best_match_chara)
                res_folders[best_match_chara]=target_folder
                if not target_folder.exists():
                    os.makedirs(target_folder)
                target_f = target_folder.joinpath(str(len(list(target_folder.iterdir())))+".jpg")
                img.save(target_f.resolve())
        return res_folders
    def dict_to_tuples(self,d:dict):
        '''
        convert inference dict to a list of tuples (tag,score), sorted by score
        :param d: pred result from dd
        :return: a list of tuples
        '''
        tuples = sorted(list(zip(d.keys(), d.values())), key=lambda x: x[1],reverse=True)
        return tuples
    def get_toi_from_dict(self,d:dict,toi:set):
        '''
        Takes in a dict and a set of tags of interest
        get the dict entries with tags of interest
        '''
        intersection = d.keys()&toi

        res_dict = {x:d[x] for x in intersection}
        return res_dict
    def cos_sim(self,d1:dict,d2:dict):
        '''
        deprecated.
        Take in two dicts
        :param d1:
        :param d2:
        :return:
        '''
        n = 0
        da =0
        db = 0
        '''
        similarity formula:
        sum(aixbi)/( sqrt(sum(a^2)) x sqrt(sum(b^2))
        '''
        print("D1 size:",len(d1),'D2 size:',len(d2))
        print(d1)
        print(d2)
        terms = set(d1).union(d2)
        '''
        d1:red,green 
        d2:blue,red 
        union: {red,green,blue}
        {1,1,0}
        {1,0,1}
        '''
        print(terms)
        dotprod = sum( d1.get(k,0.0) * d2.get(k,0.0) for k in terms  )
        magA = math.sqrt(sum(d1.get(k, 0)**2 for k in terms))
        magB = math.sqrt(sum(d1.get(k, 0) ** 2 for k in terms))
        return dotprod/(magA*magB)