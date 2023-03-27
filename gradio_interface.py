import tempfile
import cv2
import gradio as gr
from pathlib import Path
import os,shutil
import json
from PIL import Image
from io import BytesIO
import numpy as np
from datetime import datetime
from extractor import RefExtractor
import subprocess
import json
from tqdm import tqdm
from argparse import ArgumentParser
class gradio_ui(object):
    def __init__(self,args):
        self.args = args
        self.refextractor = RefExtractor(Path(args.model_dir))
        charas_path = Path("characters.json")
        if charas_path.exists():
            with open(charas_path,'r') as f:
                try:
                    self.refextractor.tagger.chara_tags = json.load(f)
                except:
                    pass
        self.last_folder = Path("output")
        self.last_mark_folder = Path("output").joinpath("mark_characters")
        self.chara_folders = {}
    #character related
    def infer_chara(self,img:np.ndarray,existing_tags:list[str],tagging_threshold:float):
        '''
        :param img: ref img to be tagged
        :param existing_tags:existing tags
        :return: update tags of a character
        '''
        if img is None:
            return gr.CheckboxGroup.update()
        self.refextractor.tagger.d.threshold = tagging_threshold
        tags:set =  self.refextractor.tagger.tag_chara(img)
        print("Existing tags:",existing_tags)
        final_tags =  list(tags.union(existing_tags))
        return gr.CheckboxGroup.update(choices=final_tags,value=final_tags,interactive=True)
    def save_chara(self,name:str,tags:list[str]):
        '''

        :param name: character name
        :param tags: selected values of character tags
        :return: update dropdown value with chara_tags keys,clear
        '''
        self.refextractor.tagger.chara_tags[name]=tags
        with open("characters.json",'w') as f:
            json.dump(self.refextractor.tagger.chara_tags,f)

        return gr.Dropdown.update(choices=list(self.refextractor.tagger.chara_tags.keys()),value=name,interactive=True),\
                gr.Radio.update(choices=list(self.refextractor.tagger.chara_tags.keys()),interactive=True)
    def erase_chara(self,name:str):
        '''

        :param name: character name
        :param tags: selected values of character tags
        :return: update dropdown value with chara_tags keys,clear
        '''
        if name in self.refextractor.tagger.chara_tags.keys():
            self.refextractor.tagger.chara_tags.pop(name)
            print (self.refextractor.tagger.chara_tags)
        if name in self.chara_folders.keys():
            self.chara_folders.pop(name)
        with open("characters.json",'w') as f:
            json.dump(self.refextractor.tagger.chara_tags,f,indent=True)


        return gr.Dropdown.update(choices=list(self.refextractor.tagger.chara_tags.keys()),value=None,interactive=True),\
                gr.Radio.update(choices=list(self.refextractor.tagger.chara_tags.keys()),interactive=True), \
                gr.Radio.update(choices=list(self.chara_folders.keys()))
    def switch_chara(self, name):
        '''

        :param name: dropdown select value
        :return: tags
        '''
        print("Chracter ",name," selected")
        if name not in self.refextractor.tagger.chara_tags.keys():
            return gr.CheckboxGroup.update(choices=[],value=[],interactive=True)
        tags = self.refextractor.tagger.chara_tags[name]
        return gr.CheckboxGroup.update(choices=tags,value=tags,interactive=True)
    # def mark_chara(self,files:list,target_charas:list[str],similarity_threshold):
    def mark_chara(self, files: list, target_charas:list, similarity_threshold):
        # files will be a list of tempfile
        print("Starting marking characters!")
        imgs = []
        for f in files:
            imgs.append(Image.open(f.name))
        # self.refextractor.tagger.mark_chara(folder_path,target_charas,similarity_threshold)
        output_folder = Path("output").joinpath("mark_characters").joinpath(datetime.now().__str__().replace(":",""))
        res_folders = self.refextractor.tagger.mark_chara_from_imgs(imgs=imgs,charas=target_charas,output_folder=output_folder,similarity_threshold=similarity_threshold)
        self.chara_folders = res_folders
        chara_folder:Path = res_folders[target_charas[0]]

        return gr.Textbox.update(value="Successfully classified characters!"),\
            gr.Gallery.update(value=[f.resolve().__str__() for f in list(chara_folder.iterdir())],label=target_charas[0],visible=True),\
            gr.Radio.update(choices=list(self.chara_folders.keys()),value=target_charas[0],visible=True,interactive=True)
    def view_mark_chara(self,target_chara:str):
        '''
        Change mark chara res gallery to target chara
        '''
        print("Viewing ",target_chara)
        if target_chara in self.chara_folders:
            chara_folder = self.chara_folders[target_chara]
            return gr.Gallery.update(value=[f.resolve().__str__() for f in list(chara_folder.iterdir())],label=target_chara,visible=True)
        return gr.Gallery.update()
    def view_last_folder(self):
        subprocess.Popen(f"explorer {self.last_folder.resolve()}")
    def send_last_to_mark(self):
        if self.last_folder.is_dir():
            fs = list(self.last_folder.iterdir())
            return gr.File.update(value=[f.resolve().__str__() for f in fs]), gr.Tabs.update(selected=1)
        return gr.File.update(), gr.Tabs.update()
    def extract_ref(self,format,mode,model_name,video_path,threshold,padding,conf_threshold,iou_threshold,min_bbox_size):
        print(video_path)
        if model_name not in self.refextractor.model_path.__str__():
            self.refextractor.model = None
            self.refextractor.model_path = Path("models/yolov8").joinpath(model_name+".pt" if ".pt" not in model_name else model_name)
        print(f"Extracting reference in {format} format, {mode} mode")
        if format=="imgs":

            res_folder_path = self.refextractor.extract_chara(video_path=video_path,output_format=format,mode=mode,frame_diff_threshold=threshold,padding=padding,conf_threshold=conf_threshold,iou_threshold=iou_threshold,min_bbox_size=min_bbox_size)
            res_imgs_paths = list(res_folder_path.iterdir())
            print(res_imgs_paths)
            self.last_folder = res_folder_path
            return gr.Gallery.update(value=[x.resolve().__str__() for x in res_imgs_paths],visible=True),\
                    gr.Video.update(visible=False),\
                    gr.Button.update(visible=True,interactive=True),\
                    gr.Button.update(visible=True,interactive=True),\
                    gr.Textbox.update(value=f"Results saved in {res_folder_path.resolve()}")
        elif format=="video":
            res_video_path = self.refextractor.extract_chara(video_path=video_path,output_format=format,mode=mode,frame_diff_threshold=threshold,padding=padding,conf_threshold=conf_threshold,iou_threshold=iou_threshold)
            print(res_video_path)
            self.last_folder = res_video_path.parent
            return gr.Gallery.update(visible=False), \
                gr.Video.update(value=res_video_path.resolve().__str__(), visible=True),\
                    gr.Button.update(visible=True,interactive=True),\
                    gr.Button.update(visible=True,interactive=True), \
                gr.Textbox.update(value=f"Result saved in {res_video_path.resolve()}")
        return gr.Gallery.update(),gr.Video.update(),gr.Button().update(),gr.Button.update(),gr.Textbox.update(value="Failed to detect characters.")
    def change_tab(self):
        print("Go to mark character tab")
        return gr.Tabs.update(selected=1)#go to mark character tab
    def mode_options(self,output_format):
        if output_format=="video":
            return gr.Radio.update(interactive=False)
        elif output_format=="imgs":
            return  gr.Radio.update(interactive=True)

        return gr.Radio.update()

    #postprocessing
    def make_grids(self,files:list,row,col,size,progress=gr.Progress()):
        output_folder = Path("output").joinpath("postprocess").joinpath("grids") .joinpath(datetime.now().__str__().replace(":", ""))
        if not output_folder.exists():
            os.makedirs(output_folder)
        row = int(row)
        col = int(col)
        size = int(size)
        imgs = [ ]
        grids = [ ]
        for f in progress.tqdm(files):
            print(f.name)
            img = cv2.imread(f.name)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            # cv2.imshow("test",img)
            # cv2.waitKey(-1)
            img = self.refextractor.pad_image(img)
            img = cv2.resize(img,(size,size),cv2.INTER_CUBIC)
            imgs.append(img)

        group_size = row*col
        for i in range(0,len(imgs),group_size):
           chunk = imgs[i:min(i+group_size,len(imgs))]
           grid = self.refextractor.make_grid(chunk,row,col)
           cv2.imwrite(output_folder.joinpath(str(len(list(output_folder.iterdir()))) + ".jpg").resolve().__str__(),grid )
           grid = cv2.cvtColor(grid,cv2.COLOR_BGR2RGB)
           grids.append(grid)

        return gr.Gallery.update(value=grids,visible=True)
    def line_option(self,selected:list):
        # "Gaussian","Laplacian","Neural Network"
        print("Selected:",selected)
        res = []
        if "Gaussian" in selected:
            res = res+[gr.Slider.update(visible=True,interactive=True),gr.Slider.update(visible=True,interactive=True),gr.Slider.update(visible=True,interactive=True)]
        else:
            res = res + [gr.Slider.update(visible=False, interactive=False),
                         gr.Slider.update(visible=False, interactive=False),
                         gr.Slider.update(visible=False, interactive=False)]
        if "Laplacian" in selected:
            res = res+[gr.Slider.update(visible=True,interactive=True)]
        else:
            res = res + [gr.Slider.update(visible=False, interactive=False)]
        if "Neural Network" in selected:
            res = res + [gr.Radio.update(visible=True, interactive=True)]
        else:
            res = res + [gr.Radio.update(visible=False, interactive=False)]
        print(len(res))
        return res
    def extract_lineart(self,files:list,selected:list[str],
                        dilate_it,dilate_ksize,gaussian_ksize,
                        laplacian_ksize,
                        nn_choice:str,
                        progress=gr.Progress()):
        '''
            line_folder_upload,
           line_process_options,
           line_gaussian_dilate_it, line_gaussian_dilate_ksize, line_gaussian_blur_ksize,
           line_laplacian_ksize,
           line_nn_choice
        '''
        output_folder = Path("output").joinpath("postprocess").joinpath("lineart").joinpath(
            datetime.now().__str__().replace(":", ""))
        if not output_folder.exists():
            os.makedirs(output_folder)

        if "Gaussian" in selected:
            self.refextractor.line_extractor.ksize_gaussian = gaussian_ksize
            self.refextractor.line_extractor.ksize_dilate = dilate_ksize
            self.refextractor.line_extractor.it_dilate = dilate_it
        if "Laplacian" in selected:
            self.refextractor.line_extractor.ksize_laplacian = laplacian_ksize



        lines = []
        for f in progress.tqdm(files):
            print(f.name)
            img = cv2.imread(f.name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if "Gaussian" in selected:
                print(img.shape)
                img = self.refextractor.line_extractor.gaussian(img)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            if "Laplacian" in selected:
                print(img.shape)
                img = self.refextractor.line_extractor.laplacian(img)
            if "Neural Network" in selected:# ["Anime2Sketch", "MangaLineExtraction"]
                if nn_choice=="Anime2Sketch":
                    print(img.shape)
                    img = self.refextractor.line_extractor.sketch_line(img)
                elif nn_choice=="MangaLineExtraction":
                    print(img.shape)
                    img = self.refextractor.line_extractor.manga_line_batch(img)
            # if img.
            img = cv2.normalize(img,img,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
            cv2.imwrite(output_folder.joinpath(str(len(list(output_folder.iterdir()))) + ".jpg").resolve().__str__(),
                        img)
            lines.append(img)
            # line = self.refextractor.lineart(img,int(dilate_it),int(dilate_ksize),int(gaussian_ksize))
            # cv2.imwrite(output_folder.joinpath(str(len(list(output_folder.iterdir()))) + ".jpg").resolve().__str__(),
            #             line)
            # lines.append(line)
        return gr.Gallery.update(value=lines, visible=True)
    def upscale(self,files:list,upscale_scale:float,upscale_model:str,upscale_sharpen:bool,upscale_sharpen_mode:str,upscale_sharpen_ksize:float,progress=gr.Progress()):
        output_folder = Path("output").joinpath("postprocess").joinpath("upscale").joinpath(
            datetime.now().__str__().replace(":", ""))
        if not output_folder.exists():
            os.makedirs(output_folder)
        res = []
        for f in progress.tqdm(files):
            print(f.name)
            # img = cv2.imread(f.name)
            img_path = output_folder.joinpath(str(len(list(output_folder.iterdir()))) + ".jpg").resolve()
            res_path = self.refextractor.upscaler.upscale(f.name,img_path,int(upscale_scale),upscale_model)
            # res_img = self.refextractor.upscaler.upscale_img(img,int(upscale_scale),upscale_model)
            if upscale_sharpen:
                res_img = cv2.imread(res_path.resolve().__str__())
                res_img = self.refextractor.upscaler.sharpen(res_img,upscale_sharpen_mode,upscale_sharpen_ksize)
                # res_img = cv2.cvtColor(res_img,cv2.COLOR_BGR2RGB)
                cv2.imwrite(res_path.resolve().__str__(),res_img)
            # res_img = cv2.cvtColor(res_img,cv2.COLOR_BGR2RGB)
            # res.append(res_img)
            res.append(res_path.resolve().__str__())
            # this has to update iteratively since this is kinda slow
        return res


        # print("Done upscaling")
        # return gr.Gallery.update(value=res,visible=True)
    def interface(self):
        output_format = gr.Radio(choices=["imgs","video"],
                                 value="imgs",
                                 label="Output format",
                                 info="If chosen 'imgs', the program will output a series of images. If chosen 'video', the program will output a video.",
                                 interactive=True)
        output_mode = gr.Radio(choices=["crop","draw","highlight"],
                               value="crop",
                               label="Output annotation mode",
                               info="If chosen 'crop', the program will output cropped images according to inference. If chosen 'draw' and 'highlight', the program will either draw or highlight the marked area on the original image.",
                               interactive=True)
        model_selection = gr.Dropdown(choices=self.refextractor.models,
                                      value=self.refextractor.models[0],
                                      label="Model",
                                      info="Which detection model to use. Models' sizes go from n->s->m->l. The larger the more accurate, but also slower.",
                                      interactive=True)
        threshold_slider = gr.Slider(minimum=0.0,maximum=1.0,value=0.2,step=0.05,label="Keyframe Threshold",info="Larger value means fewer keyframe extracted for imgs mode",interactive=True)
        padding_slider = gr.Slider(minimum=-0.5,maximum=1.0,value=0.0,step=0.05,label="Detection Padding",info="Pad the detection boxes(optional)",interactive=True)
        conf_threshold_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.1,step=0.05, label="Detection Confidence Threshold",
                                     info="How confident the detection result has to be to be considered.", interactive=True)
        iou_threshold_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.05,
                                          label="Box Merging(IOU) Threshold",
                                          info="How large the intersection of boxes has to be to be merged.",
                                          interactive=True)
        min_bbox_size_slider = gr.Number(label="Minimum Bounding Box Size",
                                         info="Detection result with shorter edge smaller than this size will not be considered.",
                                         value=300,precision=10)
        vid_upload = gr.Video(label="Upload your video!")
        vid_submit = gr.Button(value="Submit video!",variant="primary")
        vid_message = gr.Textbox(interactive=False)
        test_btn = gr.Button(value="Test")

        res_imgs = gr.Gallery(label="Result",visible=False,interactive=False)
        res_imgs.style(grid=6,container=True)

        res_vid = gr.Video(label="Result",visible=False,interactive=False)

        res_view_btn = gr.Button(value="View it in your folder 🗀",visible=True,interactive=True)
        res_send_to_mark_btn = gr.Button(value="Send to mark character",visible=True,interactive=False)
        # mark character part
        mark_use_last_folder_btn =gr.Button(value="Read last result",interactive=True)
        mark_folder_upload = gr.File(label="Upload dataset",
                                     file_count="directory",
                                     info="Upload the folder with images you want to mark",
                                     interactive=True,elem_id="mark-files"
                                     )
        mark_chara_target_selection = gr.CheckboxGroup(choices=list(self.refextractor.tagger.chara_tags.keys()),value=None,
                                                       label="Target Characters",
                                                       info="Check the names of characters you want to mark in this dataset",
                                                       # interactive=True
                                                       )
        mark_chara_similarity_threshold = gr.Slider(minimum=0.0,maximum=1.0,value=0.4,step=0.05,
                                                    label="Similarity threshold",
                                                    info="How similar the image has to be to be considered as a character.",
                                                    interactive=True
                                                    )
        mark_chara_res_chara_selection = gr.Radio(label="Show Result of",visible=False,interactive=False)
        mark_chara_res_gallery = gr.Gallery(label="Result",visible=False)
        mark_chara_res_gallery.style(grid=6,container=True)

        mark_btn = gr.Button(value="Start marking!",variant="primary",interactive=True)
        mark_message = gr.Textbox(interactive=False,label="Message")
        # character manipulation
        mark_chara_selection = gr.Dropdown(label="Character",
                                           choices=list(self.refextractor.tagger.chara_tags.keys()),
                                           interactive=True)
        mark_chara_tagging_threshold = gr.Slider(label="Tagging threshold",
                                                 value=self.refextractor.tagger.d.threshold,
                                                 minimum=0.0,maximum=1.0,step=0.05,
                                                 interactive=True)
        mark_chara_img = gr.Image(label='Character Reference Image',
                                  # tool="select",
                                  interactive=True)
        mark_chara_tags = gr.CheckboxGroup(choices=[],label='Tags')
        mark_chara_name = gr.Textbox(label='Character Name',
                                     placeholder='Put the name of your character here',
                                     info="If character name exists, this will update the character's tags",
                                     interactive=True)
        mark_chara_submit = gr.Button(value="Save Character",variant="primary",interactive=True)
        mark_chara_erase  = gr.Button(value="Delete Character", interactive=True)

        # post processing

        #make grids
        grid_rows = gr.Slider(minimum=1,maximum=12,value=3,step=1,label="Grid Rows",interactive=True)
        grid_cols = gr.Slider(minimum=1, maximum=12, value=3, step=1, label="Grid Columns",interactive=True)
        grid_size = gr.Number(value=300,label="Grid size",info="Size of each image in the grid",interactive=True)
        grid_folder_upload = gr.File(label="Upload dataset",
                                     file_count="directory",
                                     info="Upload the folder with images you want to make into grids",
                                     interactive=True,elem_id="grid-files"
                                     )
        grid_submit_btn = gr.Button(value="Make grids",variant="primary",interactive=True)
        grid_res_gallery = gr.Gallery(label="Grid Result").style(grid=6)

        #extract line art
        line_folder_upload = gr.File(label="Upload dataset",
                                     file_count="directory",
                                     info="Upload the folder with images you want to extract lineart",
                                     interactive=True,elem_id="line-files"
                                     )
        line_process_options = gr.CheckboxGroup(choices=["Gaussian","Laplacian","Neural Network"],value="Gaussian",label="Line art extraction options",
                                                info="Choosing multiple options will result in img processed through each method sequentially.",interactive=True,visible=True)
        '''
        for gaussian method
        '''
        line_gaussian_dilate_it = gr.Slider(label="Iterations",
                                            info="More iteration usually gives better result, but slower.",
                                            minimum=1, maximum=20, value=1, step=1, interactive=True)
        line_gaussian_dilate_ksize = gr.Slider(label="Dilation kernel size",
                                               info="Higher value gives more lines but also more noises.",
                                               minimum=1, maximum=15, value=3, step=2, interactive=True)
        line_gaussian_blur_ksize = gr.Slider(label="Gaussian blur kernel size",
                                             info="You can play with this.",
                                             minimum=1, maximum=15, value=3, step=2, interactive=True)
        '''
        For Laplacian method 
        '''
        line_laplacian_ksize = gr.Slider(label="Laplacian kernel size",
                                            info="Larger value yields more lines, while more noise will be taken in",
                                            minimum=1, maximum=15, value=3, step=2, interactive=True,visible=False)
        '''
        For Neural Network method
        '''
        line_nn_choice = gr.Radio(label="Neural Network Model",choices=["Anime2Sketch", "MangaLineExtraction"],value="Anime2Sketch",interactive=True,visible=False)



        line_submit_btn = gr.Button(value="Extract lineart",variant="primary",interactive=True)
        line_res_gallery = gr.Gallery(label="Lineart Result").style(grid=6)

        #Upscaling
        upscale_scale = gr.Slider(label="Scale to",
                                  minimum=1,maximum=4,step=1,value=2)
        upscale_model = gr.Dropdown(label="Model",
                                    choices=self.refextractor.upscaler.models,
                                    value=self.refextractor.upscaler.models[0],
                                    interactive=True)
        upscale_folder_upload = gr.File(label="Upload dataset",
                                     file_count="directory",
                                     info="Upload the folder with images you want to upscale",
                                     interactive=True,elem_id="up-files"
                                     )
        upscale_sharpen = gr.Checkbox(label="Sharpen",
                                      value=False,
                                      info="Sharpen the images after upscaling",
                                      interactive=True)
        upscale_sharpen_mode = gr.Radio(label="Sharpen Mode",
                                        value="Laplace",
                                        choices=self.refextractor.upscaler.sharpen_modes,
                                        interactive=True)
        upscale_sharpen_ksize = gr.Number(label="Kernal Size",
                                          info="This only works for USM mode, higher means stronger effect.",
                                          value=1,
                                          interactive=True)
        upscale_submit_btn = gr.Button(value="Upscale images!", variant="primary", interactive=True)
        upscale_stop_btn = gr.Button(value="Stop",interactive=True)
        upscale_res_gallery = gr.Gallery(label="Upscale Result").style(grid=6)

        with gr.Blocks(title="AniRef",css="""
            .file-preview{
                max-height:20vh;
                overflow:scroll !important;
            }
        """) as demo:
            with gr.Tabs() as tabs:
                with gr.TabItem("Inference",id=0):
                    with gr.Row(variant="compact"):
                        output_format.render()
                        output_mode.render()
                        model_selection.render()
                    with gr.Row():
                        with gr.Accordion(label="Advanced",open=False):
                            conf_threshold_slider.render()
                            iou_threshold_slider.render()
                            threshold_slider.render()
                            padding_slider.render()
                            min_bbox_size_slider.render()
                    with gr.Row():
                        vid_upload.render()
                    with gr.Row():
                        vid_submit.render()
                        # test_btn.render()
                    with gr.Row():
                        with gr.Column():
                        # with gr.Row():
                            with gr.Row():
                                vid_message.render()
                            with gr.Row():
                                res_imgs.render()
                            with gr.Row():
                                res_vid.render()
                            with gr.Row():
                                res_view_btn.render()
                                res_send_to_mark_btn.render()
                with gr.TabItem("Mark Characters",id=1):
                    with gr.Row():
                        with gr.Column(scale=3):
                            mark_use_last_folder_btn.render()
                            mark_chara_target_selection.render()
                            mark_chara_similarity_threshold.render()
                            mark_btn.render()
                            mark_message.render()
                            mark_chara_res_chara_selection.render()
                            mark_chara_res_gallery.render()
                            mark_folder_upload.render()
                        with gr.Column(scale=1):
                            mark_chara_selection.render()
                            mark_chara_tagging_threshold.render()
                            mark_chara_img.render()
                            mark_chara_tags.render()
                            mark_chara_name.render()
                            mark_chara_submit.render()
                            mark_chara_erase.render()
                with gr.TabItem("Postprocessing",id=2):
                    with gr.Tabs(selected=0):
                        with gr.TabItem(label="Make grids",id=0):
                            with gr.Row():
                                grid_rows.render()
                                grid_cols.render()
                            with gr.Row():
                                grid_size.render()
                            with gr.Row():
                                grid_submit_btn.render()
                            with gr.Row():
                                grid_res_gallery.render()
                            with gr.Row():
                                grid_folder_upload.render()
                        with gr.TabItem(label="Extract lineart", id=1):
                            with gr.Row():
                                line_submit_btn.render()
                            with gr.Row():
                                line_process_options.render()
                            with gr.Row():
                                line_gaussian_dilate_it.render()
                                line_gaussian_dilate_ksize.render()
                                line_gaussian_blur_ksize.render()
                            with gr.Row():
                                line_laplacian_ksize.render()
                            with gr.Row():
                                line_nn_choice.render()
                            with gr.Row():
                                line_res_gallery.render()
                            with gr.Row():
                                line_folder_upload.render()
                        with gr.TabItem(label="Upscaling", id=2):
                            with gr.Row():
                                upscale_submit_btn.render()
                                # upscale_stop_btn.render()
                            with gr.Row():
                                with gr.Box():
                                    upscale_scale.render()
                                    upscale_model.render()
                                with gr.Box():
                                    upscale_sharpen.render()
                                    upscale_sharpen_mode.render()
                                    upscale_sharpen_ksize.render()
                            with gr.Row():
                                upscale_res_gallery.render()
                            with gr.Row():
                                upscale_folder_upload.render()
            output_format.change(fn=self.mode_options,inputs=output_format,outputs=output_mode)
            vid_submit.click(fn=self.extract_ref,inputs=[output_format,output_mode,model_selection,vid_upload,threshold_slider,padding_slider,conf_threshold_slider,iou_threshold_slider,min_bbox_size_slider],outputs=[res_imgs,res_vid,res_view_btn,res_send_to_mark_btn,vid_message])
            # test_btn.click(fn=self.change_tab,inputs=None,outputs=tabs)
            res_send_to_mark_btn.click(fn=self.send_last_to_mark, outputs=[mark_folder_upload, tabs])
            res_view_btn.click(fn=self.view_last_folder)

            # character marking stuff
            mark_use_last_folder_btn.click(fn=self.send_last_to_mark,outputs=[mark_folder_upload,tabs])
            mark_chara_img.change(fn=self.infer_chara,inputs=[mark_chara_img,mark_chara_tags,mark_chara_tagging_threshold],outputs=[mark_chara_tags])
            mark_chara_submit.click(fn=self.save_chara,inputs=[mark_chara_name,mark_chara_tags],outputs=[mark_chara_selection,mark_chara_target_selection])
            mark_chara_erase.click(fn=self.erase_chara, inputs=[mark_chara_selection],
                                    outputs=[mark_chara_selection, mark_chara_target_selection, mark_chara_res_chara_selection])
            mark_chara_selection.change(fn=self.switch_chara,inputs=mark_chara_selection,outputs=mark_chara_tags)

            mark_btn.click(fn=self.mark_chara,inputs=[mark_folder_upload,mark_chara_target_selection,mark_chara_similarity_threshold],outputs=[mark_message,mark_chara_res_gallery,mark_chara_res_chara_selection])
            mark_chara_res_chara_selection.change(fn=self.view_mark_chara,inputs=[mark_chara_res_chara_selection],outputs=[mark_chara_res_gallery])

            grid_submit_btn.click(fn=self.make_grids,inputs=[grid_folder_upload,grid_rows,grid_cols,grid_size],outputs=[grid_res_gallery])
            line_submit_btn.click(fn=self.extract_lineart, inputs=[line_folder_upload,
                                                                   line_process_options,
                                                                   line_gaussian_dilate_it, line_gaussian_dilate_ksize, line_gaussian_blur_ksize,
                                                                   line_laplacian_ksize,
                                                                   line_nn_choice
                                                                   ], outputs=[line_res_gallery])
            upscale_event =  upscale_submit_btn.click(fn=self.upscale,inputs=[upscale_folder_upload,upscale_scale,upscale_model,upscale_sharpen,upscale_sharpen_mode,upscale_sharpen_ksize],outputs=[upscale_res_gallery])
            upscale_stop_btn.click(cancels=[upscale_event],fn=None)

            line_process_options.change(fn=self.line_option,inputs=[line_process_options],outputs=[line_gaussian_dilate_it,line_gaussian_dilate_ksize,line_gaussian_blur_ksize,
                                                                                                   line_laplacian_ksize,
                                                                                                   line_nn_choice])
        if args.port is None:
            demo.queue().launch(share=args.share,inbrowser=args.inbrowser)
        else:
            demo.queue().launch(share=args.share, inbrowser=args.inbrowser,server_port=args.port)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--share",action="store_true",default=False,help="Expose the web app to public.")
    parser.add_argument("--inbrowser",action="store_true",default=False,help="Open the web app in browser.")
    parser.add_argument("--port",type=int,nargs="?",help="Specify the server port, if not specified, then the app will assign a random port.")
    parser.add_argument("--model_dir",default="models",type=str,help="Specify the models folder containing yolov8 and line art models.")
    args = parser.parse_args()
    ui = gradio_ui(args)
    ui.interface()