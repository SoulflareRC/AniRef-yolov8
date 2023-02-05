import os

import cv2
import gradio
import numpy as np
from PIL import Image
import gradio as gr
import deepdanbooru_onnx as dd
import os
import subprocess
from utils.extract_frames import *

class gradio_ui(object):
    def __init__(self):
        self.extractor = Extractor(None,output_dir="temp")
        self.d = dd.DeepDanbooru()
        self.chara_tags = None
        self.chara_dict = {}

        self.current_tags = []
        #only for add tags

        self.current_chara = None

        self.kf_max = 100
        #
    # def load_scripts(self):

    def dd_config(self,threshold):
        self.d.threshold = threshold
        print('Set deepdanbooru interrogate threshold to ',self.d.threshold)
    def test(self):
        print('Hello WTF????')
    def get_ref(self,video):
        output_dir = "temp"
        cmd = f"ffmpeg -i {video} "
    def get_scene(self,start_frameCnt):
        frames = self.extractor.extract_scene(start_frameCnt)
        print(len(frames))
        imgs = [cv2.cvtColor(f.img,cv2.COLOR_BGR2RGB) for f in frames ]
        print('Scene frames: ' ,len(imgs))
        return gr.Gallery.update(value=imgs,label="Result")
        # return [f.img for f in frames ]
    def get_keyframes2(self,video):
        '''
         Do at this step:
         extract keyframes + framestep as "scene"
         (Next: select a scene
         :param video:
         :return: list of Image+caption
         '''
        print(type(video))
        print("Video input:", video)  # video passes in a path to video
        self.extractor.video = video
        frames = self.extractor.extract_IPBFrames('I')
        self.extractor.frames = frames
        output_imgs = []
        for i in range(len(frames)):
            frame = frames[i]
            print(type(frame.img))
            print(type(frame.frameCnt))
            item = (cv2.cvtColor(frame.img,cv2.COLOR_BGR2RGB),str(frame.frameCnt))
            output_imgs.append(item)
        print(len(output_imgs))
        print("Done!")
        print(len(output_imgs))
        return output_imgs  # tuple of (img,caption)
    def get_keyframes(self,video):
        '''
        Do at this step:
        extract keyframes + framestep as "scene"
        (Next: select a scene
        :param video:
        :return: list of Image
        '''
        print("WTF???")
        print(type(video))
        print("Video input:",video)#video passes in a path to video
        self.extractor.video = video
        frames = self.extractor.extract_IPBFrames('I')
        self.extractor.frames = frames
        # frames =self.extractor.frames
        output_imgs = []
        output_btn = []
        for i in range(len(frames)):
            frame = frames[i]
            print(type(frame.img))
            print(type(frame.frameCnt))
            img = gr.Image.update(value =cv2.cvtColor(frame.img,cv2.COLOR_BGR2RGB),label=frame.frameCnt,visible=True)
            img:gr.Image
            # img.set_event_trigger(fn=self.test,event_name='click')
            output_imgs.append(img)
            # output_imgs.append(gr.Image.update(value = frame.img,label="",visible=True))
            output_btn.append(gr.Button.update(value="Skip", visible=True))
        # cap = cv2.VideoCapture(video)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # frameCnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #     split into 10 for testing

        # timestamps = np.arange(0,frameCnt,frameCnt/9)
        #
        # for i in range(len(timestamps)):
        #     cap.set(cv2.CAP_PROP_POS_FRAMES,timestamps[i])
        #     ret,frame = cap.read()
        #     print(frame.shape)#h,w,c
        #     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        #     # print("Ret?",ret)
        #     # cv2.imshow(f'frame{timestamps[i]}',frame)
        #     # cv2.waitKey(-1)
        #     output_imgs.append(gr.Image.update(value=frame,label=cap.get(cv2.CAP_PROP_POS_FRAMES),visible=True))
        #     output_btn.append(gr.Button.update(value="Skip",visible=True))
        print(len(output_imgs),len(output_btn))

        while len(output_imgs)<self.kf_max:
            output_imgs.append(gr.Image.update(visible=False))
            output_btn.append(gr.Button.update(visible=False))
        print("Done!")
        print(len(output_imgs))
        print(len(output_imgs+output_btn))
        return output_imgs #+output_btn


    def add_tag_chara(self,tag:str,tags):
        print(tag)
        self.current_tags.append(tag)
        return gr.Textbox.update(value=""), gr.CheckboxGroup.update(choices=self.current_tags)
    #set current character to this for quick use
    def infer_chara(self,img:np.ndarray)->gr.CheckboxGroup:
        if img is not None:
            img = Image.fromarray(img)
            img = dd.process_image(img)
            pred_dict = self.d(img)
            tuples =sorted(list(zip(pred_dict.keys(),pred_dict.values())),key=lambda x:x[1])
            print(tuples)
            self.current_tags=[tag[0] for tag in tuples]
            return gr.CheckboxGroup.update(choices=self.current_tags,interactive=True)
        else:
            return gr.CheckboxGroup.update()
    def save_chara(self,chara_img,chara_name,chara_tags):

        self.chara_dict[chara_name] = {
            'name':chara_name,
            'tags':chara_tags,
            'img':chara_img
        }
        self.current_tags.clear()
        return gr.Image.update(value=None),\
            gr.Textbox.update(value=None),\
            gr.CheckboxGroup.update(choices=None,value=[]),\
            gr.Dropdown.update(choices=list(self.chara_dict.keys()),interactive=True)
    def switch_chara(self,chara_name):
        self.current_chara = self.chara_dict[chara_name]
        return gr.Image.update(value=self.current_chara['img']),\
               gr.Textbox.update(value=self.current_chara['name']),\
               gr.CheckboxGroup.update(choices=self.current_chara['tags'],
                                       value=self.current_chara['tags'])
    def interface(self):

        original = gradio.routes.templates.TemplateResponse
        script_path = "script.js"
        style_path = "style.css"
        with open(style_path) as f:
            css_str = f.read()
        injection_js = ""
        with open(script_path) as f:
            injection_js += f.read()
        #inject jquery
        injection_js += """<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.1/jquery.min.js"></script>"""
        # inject bootstrap
        injection_js += """<script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js" integrity="sha384-oBqDVmMz9ATKxIep9tiCxS/Z9fNfEXiDAYTujMAeBAsjFuCZSmKbSSUnQlmh/jp3" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.min.js" integrity="sha384-mQ93GR66B00ZXjt0YO5KlohRA5SY2XofN4zfuZxLkoj1gXtW8ANNCe9d5Y3eG5eD" crossorigin="anonymous"></script>"""
        # inject custom js
        injection_js += '<script type="text/javascript">'
        with open(script_path) as f:
            s = f.read()
            print(s)
            injection_js += s
        injection_js += '</script>'
        def template_response(*args, **kwargs):
            print("Hello from modified reponse")
            res = original(*args, **kwargs)#<starlette.templating._TemplateResponse object
            # print(res.body)
            res_str = res.body.decode(encoding='utf-8')
            idx = res_str.find('</body>')
            res_str = res_str[:idx]+injection_js+res_str[idx:]
            res.body = bytes(res_str,'utf-8')
            # res.body = res.body.replace(
            #     b'</head>', f'{javascript}</head>'.encode("utf8"))
            res.init_headers()
            return res
        gradio.routes.templates.TemplateResponse = template_response
        print("Css styles:",css_str)

        imgs = []
        img_dir = 'images'
        for f in os.listdir(img_dir):
            img = Image.open(os.path.join(img_dir,f))
            imgs.append(img)

        vid_input = gr.Video(label='Input video',elem_id='vid_input')
        vid_extract_keyframes = gr.Button(value='Extract Scenes')
        vid_keyframe_gallery = gr.Gallery(label="Keyframes",elem_id='kfgallery',value=imgs)
        vid_keyframe_gallery.style(grid=10,height="300px")

        vid_keyframe_set = []
        vid_keyframe_btn_set = []
        # vid_kf = gr.Image(visible=False,interactive=False)
        for i in range(self.kf_max):
            img = gr.Image(visible=False,interactive=False)

            vid_keyframe_set.append(img)
            # vid_keyframe_set.append(gr.Image(visible=False,interactive=False))
            vid_keyframe_btn_set.append(gr.Button(visible=False,interactive=False))
        vid_keyframe_gallery = gr.Gallery(label = "Keyframes")

        chara_img = gr.Image(label='Character Reference')
        chara_tags = gr.CheckboxGroup(choices=[],label='Tags')
        chara_add_tag = gr.Textbox(placeholder="Press Enter to add tag",show_label=False)
        chara_name = gr.Textbox(label='Character Name',placeholder='Put the name of your character here')
        # chara_tags_delete = gr.Button(value='Delete tags')

        chara_tags_submit = gr.Button(value='Submit tags',variant='primary')
        chara_selection = gr.Dropdown(label='Select Character')

        #settings tab
        dd_threshold = gr.Slider(minimum=0,maximum=1,value=self.d.threshold,step=0.01,interactive=True)

        #results tab
        result_img_settings = gr.CheckboxGroup(label="Image options",choices=["Clean Background","Crop Character","Line Art Only"])
        result_vid_settings = gr.Radio(label="Extraction options",choices=["Images","Video Clips"])
        result_frameCnt = gr.Number(value=0,label="Starting frame number")
        result_btn = gr.Button(value = "Extract Reference!")
        result_gallery  = gr.Gallery(label="Keyframes",elem_id='resgallery')

        css_style="""
        #kfs-wrap{
            overflow:scroll;
            flex-wrap:nowrap;
            flex-direction:row;
        }
        ::-webkit-scrollbar-corner,::-webkit-scrollbar-track-piece,::-webkit-scrollbar-button,::-webkit-scrollbar-track {
            background-color:transparent;
        }
        ::-webkit-scrollbar-thumb {
            background-color:rgba(255,255,255,0.5);
            border:1px black solid;
            border-radius:50%;
            transition:0.5s;
        }
        ::-webkit-scrollbar-thumb:hover{
            background-color:rgba(255,255,255,0.8);
        }
        """
        with gr.Blocks(css=css_style) as demo:
            with gr.Tab('Video'):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Column():
                            result_img_settings.render()
                            result_vid_settings.render()
                            result_frameCnt.render()
                            result_btn.render()
                            result_gallery.render()
                    with gr.Column(scale=1):
                        with gr.Tab("Video"):
                            with gr.Column(scale=3):
                                vid_input.render()
                                vid_extract_keyframes.render()
                                # with gr.Row(elem_id="kfs-wrap") as r:
                                #     # r.add(vid_kf)
                                #     # vid_kf.render()
                                #     for i in range(self.kf_max):
                                #         with gr.Column() as c:
                                #             vid_keyframe_set[i].render()
                                #             vid_keyframe_set[i].set_event_trigger(fn=self.test,event_name='click',inputs=None,outputs=None)
                                #             vid_keyframe_btn_set[i].render()
                                vid_keyframe_gallery.render()
                        with gr.Tab("Character"):
                            with gr.Column():
                                chara_selection.render()
                                chara_img.render()
                                chara_tags.render()
                                chara_add_tag.render()
                                chara_name.render()
                                with gr.Row():
                                    chara_tags_submit.render()
                            # chara_tags_delete.render()
            with gr.Tab("Settings"):
                dd_threshold.render()

            chara_img.change(fn=self.infer_chara,inputs=chara_img,outputs=chara_tags)
            chara_selection.change(fn=self.switch_chara,inputs = chara_selection,outputs=[chara_img,chara_name,chara_tags])
            chara_add_tag.submit(fn=self.add_tag_chara,inputs=[chara_add_tag,chara_tags],outputs=[chara_add_tag,chara_tags])
            # chara_tags_delete.click(fn=self.delete_chara_tags,inputs=chara_tags,outputs=chara_tags)
            chara_tags_submit.click(fn=self.save_chara,inputs = [chara_img,chara_name,chara_tags], outputs=[chara_img,chara_name,chara_tags,chara_selection])

            # vid_extract_keyframes.click(fn=self.get_keyframes,inputs=vid_input,outputs=vid_keyframe_set)#+vid_keyframe_btn_set)
            vid_extract_keyframes.click(fn=self.get_keyframes2, inputs=vid_input,
                                        outputs=vid_keyframe_gallery)  # +vid_keyframe_btn_set)

            # vid_extract_keyframes.click(fn=self.test,inputs=[],outputs=[])

            #settings tab
            dd_threshold.change(fn=self.dd_config,inputs=dd_threshold)

            result_btn.click(fn=self.get_scene,inputs=result_frameCnt,outputs=result_gallery)

        demo.launch(server_port=3000)

        return demo
# if __name__ == '__main__':
print("Hey!")
demo = gradio_ui().interface()
# demo.launch(server_port=1000)