import PIL.Image
from deepdanbooru_onnx import DeepDanbooru,process_image
import math
import numpy as np
import cv2
from PIL import  Image
from collections import Counter

class Tagger(object):
    '''
    Wrapper class for deepdanbooru utility
    '''
    def __init__(self):
        self.d = DeepDanbooru(threshold=0.5)
        self.threshold_s = 0.4
    def __call__(self,img:Image.Image):
        '''
        Wrapped call of deepdanbooru tagger
        :param img:
        :return: a dict
        '''
        img = process_image(img)
        return self.d(img)
    def dict_to_tuples(self,d:dict):
        '''
        convert inference dict to a list of tuples (tag,score), sorted by score
        :param d: pred result from dd
        :return: a list of tuples
        '''
        tuples = sorted(list(zip(d.keys(), d.values())), key=lambda x: x[1],reverse=True)
        return tuples
    def cos_sim(self,d1:dict,d2:dict):
        '''
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