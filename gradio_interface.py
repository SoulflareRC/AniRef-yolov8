import tempfile

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
class gradio_ui(object):
    def __init__(self):
        self.refextractor = RefExtractor()

        self.last_folder = Path("output")
    #character related
    def infer_chara(self,img:np.ndarray,existing_tags:list[str]):
        '''
        :param img: ref img to be tagged
        :param existing_tags:existing tags
        :return: update tags of a character
        '''
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
        return gr.Dropdown.update(choices=list(self.refextractor.tagger.chara_tags.keys()),value=name,interactive=True),\
                gr.CheckboxGroup.update(choices=list(self.refextractor.tagger.chara_tags.keys()),interactive=True)

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
    def mark_chara(self,files:list,target_charas:list[str],similarity_threshold):
        # files will be a list of tempfile
        print("Starting marking characters!")
        imgs = []
        for f in files:
            imgs.append(Image.open(f.name))
        # self.refextractor.tagger.mark_chara(folder_path,target_charas,similarity_threshold)
        output_folder = Path("output").joinpath("mark_characters").joinpath(datetime.now().__str__().replace(":",""))
        self.refextractor.tagger.mark_chara_from_imgs(imgs=imgs,charas=target_charas,output_folder=output_folder)
        return gr.Textbox.update(value="Successfully classified characters!")
    def view_last_folder(self):
        subprocess.Popen(f"explorer {self.last_folder.resolve()}")
    def extract_ref(self,format,mode,model_name,video_path,threshold,padding):
        if model_name not in self.refextractor.model_path.__str__():
            self.refextractor.model = None
            self.refextractor.model_path = Path("models/yolov8").joinpath(model_name+".pt" if ".pt" not in model_name else model_name)
        print(f"Extracting reference in {format} format, {mode} mode")
        if format=="imgs":

            res_folder_path = self.refextractor.extract_chara(video_path=video_path,output_format=format,mode=mode,frame_diff_threshold=threshold,padding=padding)
            res_imgs_paths = list(res_folder_path.iterdir())
            print(res_imgs_paths)
            self.last_folder = res_folder_path
            return gr.Gallery.update(value=[x.resolve().__str__() for x in res_imgs_paths],visible=True),\
                    gr.Video.update(visible=False),\
                    gr.Button(visible=True,interactive=True)
        elif format=="video":
            res_video_path = self.refextractor.extract_chara(video_path=video_path,output_format=format,mode=mode,frame_diff_threshold=threshold,padding=padding)
            print(res_video_path)
            self.last_folder = res_video_path.parent
            return gr.Gallery.update(visible=False), \
                gr.Video.update(value=res_video_path.resolve().__str__(), visible=True),\
                    gr.Button(visible=True,interactive=True)
        return gr.Gallery.update(),gr.Video.update(),gr.Button()
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
        threshold_slider = gr.Slider(minimum=0.0,maximum=1.0,value=0.2,label="Keyframe Threshold",info="Larger value means fewer keyframe extracted for imgs mode",interactive=True)
        padding_slider = gr.Slider(minimum=-0.5,maximum=1.0,value=0.0,label="Detection Padding",info="Pad the detection boxes(optional)",interactive=True)

        vid_upload = gr.Video(label="Upload your video!")
        vid_submit = gr.Button(value="Submit video!",variant="primary")
        # test_btn = gr.Button(value="Test")

        res_imgs = gr.Gallery(label="Result",visible=False,interactive=False)
        res_imgs.style(grid=6,container=True)

        res_vid = gr.Video(label="Result",visible=False,interactive=False)

        res_view_btn = gr.Button(value="View it in your folder 🗀",visible=False,interactive=True)

        # mark character part
        mark_use_last_folder_btn =gr.Button(value="Read last result",interactive=True)
        mark_folder_upload = gr.File(label="Upload dataset",
                                     file_count="directory",
                                     info="Upload the folder with images you want to mark",
                                     interactive=True
                                     )
        mark_chara_target_selection = gr.CheckboxGroup(choices=[],value=None,
                                                       label="Target Characters",
                                                       info="Check the names of characters you want to mark in this dataset",
                                                       # interactive=True
                                                       )
        mark_chara_similarity_threshold = gr.Slider(minimum=0.0,maximum=1.0,value=0.4,
                                                    label="Similarity threshold",
                                                    info="How similar the image has to be to be considered as a character.",
                                                    interactive=True
                                                    )
        mark_btn = gr.Button(value="Start marking!",variant="primary",interactive=True)
        mark_message = gr.Textbox(interactive=False,label="Message")
        # character manipulation
        mark_chara_selection = gr.Dropdown(label="Character",
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

        with gr.Blocks(title="AniRef") as demo:
            with gr.Tab("Inference"):
                with gr.Row(variant="compact"):
                    output_format.render()
                    output_mode.render()
                    model_selection.render()
                with gr.Row():
                    with gr.Accordion(label="Advanced",open=False):
                        threshold_slider.render()
                        padding_slider.render()
                with gr.Row():
                    vid_upload.render()
                with gr.Row():
                    vid_submit.render()
                    # test_btn.render()
                with gr.Row():
                    # with gr.Row():
                    res_imgs.render()
                    res_vid.render()
                    res_view_btn.render()
            with gr.Tab("Mark Characters") as mark_tab:
                with gr.Row():
                    with gr.Column(scale=3):
                        mark_use_last_folder_btn.render()
                        mark_chara_target_selection.render()
                        mark_chara_similarity_threshold.render()
                        mark_folder_upload.render()
                        mark_btn.render()
                        mark_message.render()
                    with gr.Column(scale=1):
                        mark_chara_selection.render()
                        mark_chara_img.render()
                        mark_chara_tags.render()
                        mark_chara_name.render()
                        mark_chara_submit.render()
            vid_submit.click(fn=self.extract_ref,inputs=[output_format,output_mode,model_selection,vid_upload,threshold_slider,padding_slider],outputs=[res_imgs,res_vid,res_view_btn])
            # test_btn.click(fn=tab_hist.select)
            res_view_btn.click(fn=self.view_last_folder)

            # character marking stuff
            mark_chara_img.change(fn=self.infer_chara,inputs=[mark_chara_img,mark_chara_tags],outputs=[mark_chara_tags])
            mark_chara_submit.click(fn=self.save_chara,inputs=[mark_chara_name,mark_chara_tags],outputs=[mark_chara_selection,mark_chara_target_selection])
            mark_chara_selection.change(fn=self.switch_chara,inputs=mark_chara_selection,outputs=mark_chara_tags)

            mark_btn.click(fn=self.mark_chara,inputs=[mark_folder_upload,mark_chara_target_selection,mark_chara_similarity_threshold],outputs=mark_message)
        demo.launch(debug=True)
ui = gradio_ui()
ui.interface()