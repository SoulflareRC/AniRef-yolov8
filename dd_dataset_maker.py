import os
import pathlib
from pathlib import Path
import shutil
from PIL import Image
import hashlib
import sqlite3
'''
SQLite databse file format:
posts
├── id (INTEGER)
├── md5 (TEXT)
├── file_ext (TEXT)
├── tag_string (TEXT)
└── tag_count_general (INTEGER)
All of image files are located in sub-folder which named first 2 characters of its filename.
'''
tag_count_general = 12
#this is important because badly tagged image will deteriorate the training effect.
def get_md5(img:Image.Image):
    return hashlib.md5(img.tobytes()).hexdigest()
def clean_tags(tags:set,tags_in:set)->list:
    # clean tags in dataset,only use the intersection of tags and tags_in
    return list(tags.intersection(tags_in))


def make_dataset_synthesis(dataset_dir,output_dir):
    '''
    make dataset from generated images
    parameters are in the first line of the info of the generated images
    :param dataset_dir:
    :param output_dir:
    :return:
    '''
    if not os.path.exists(dataset_dir):
        print("Dataset dir doesn't exist!")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    dataset = Path(dataset_dir)
    output = Path(output_dir)
    output_images = output.joinpath("images")

    dbname = 'my-dataset.sqlite'
    sqldb = output.joinpath(dbname)#sqlite database file
    create_database(sqldb.resolve())
    images = list(dataset.iterdir())
    idx = 0
    for f in images:
        if f.is_file():
            img = Image.open(f.resolve())
            md5 = get_md5(img)
            prefix = md5[:2]
            # parent dir must be the first 2 chars of the image
            target_dir = output_images.joinpath(prefix)
            if not target_dir.exists():
                os.makedirs(target_dir,exist_ok=True)
            target_file = target_dir.joinpath(f.name)
            shutil.copy(f,target_file)#hmm why didn't pathlib have a copy()?
            parameters = img.info['parameters'].split('\n')
            tags = parameters[0].split(',')
            print(f"Image {f.name} has {len(tags)} tags")
            #info list should still in the form of [id(this should not matter?),md5(this only has to be some unique name,doesn't actually have to be md5),ext,tag_string,tag_count)
            info = [idx,md5,f.suffix.replace('.',''),' '.join(tags),tag_count_general]
            print(info)
            insert_row(sqldb.resolve(), info)
            idx+=1
def make_dataset(dataset_dir,output_dir,score_threshold = 0):
    '''
    1.jpg
    1.txt

    :param dataset_dir: make dataset from dataset_dir. Dataset_dir must contain images(filename is md5) and their corresponding text files(same filename but ext is txt)
    :param output_dir:
    :return:
    '''
    if not os.path.exists(dataset_dir):
        print("Dataset dir doesn't exist!")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir,exist_ok=True)
    dataset = Path(dataset_dir)
    output = Path(output_dir)
    output_images = output.joinpath("images")

    dbname = 'my-dataset.sqlite'
    sqldb = output.joinpath(dbname)#sqlite database file
    create_database(sqldb.resolve())

    image_list = list(dataset.iterdir())#list all children

    #find all images
    suffixes = ('.jpg','.png','.bmp')
    for f in image_list:
        if f.suffix in suffixes:
            #is image,find its tag file
            md5 = f.stem
            image = f.resolve()#get absolute path
            text = f.parent.joinpath(f"{md5}.txt")
            print(image)
            print(text)
            prefix = md5[:2]
            # parent dir must be the first 2 chars of the image
            target_dir = output_images.joinpath(prefix)
            if not target_dir.exists():
                os.makedirs(target_dir,exist_ok=True)
            target_file = target_dir.joinpath(f.name)
            shutil.copy(f,target_file)#hmm why didn't pathlib have a copy()?

            #insert data of this image into db file
            with open(text.resolve()) as f:
                info = f.read().split(',')
                if score_threshold>0 and len(info)>4:
                    if int(info[4])<score_threshold:
                        continue
                info[0]=int(info[0])
                info.insert(4,tag_count_general)
                # this is [id,md5,tag_string,ext,tag_count] [score]
                # should be [id,md5,ext,tag_string,tag_count] (5 args to pass into insert_row)
                info[2],info[3] = info[3],info[2]
                print(info)
                insert_row(sqldb.resolve(),info)
def create_database(name:str):
    # for creating sqlite db file
    db = sqlite3.connect(name)
    c = db.cursor()
    c.execute('''
            CREATE TABLE posts
            (id     INT PRIMARY KEY NOT NULL,
             md5    TEXT            NOT NULL,
             file_ext   TEXT        NOT NULL,
             tag_string TEXT        NOT NULL,
             tag_count_general  INT NOT NULL 
            );
    ''')
    db.commit()
    db.close()
def insert_row(name:str,values:list):
    '''
    :param name: database name
    :param values:should contain [id,md5,ext,tags,tag_count] in order
    '''
    values_str = ""
    values_parsed = []
    for item in values:
        if type(item)==str:
            values_parsed.append( "'"+item+"'" )
        elif type(item)==int:
            values_parsed.append(str(item))
    values_str = ','.join(values_parsed)
    print(values_str)
    db = sqlite3.connect(name)
    c = db.cursor()
    c.execute(f'''
        INSERT INTO posts values ({values_str});
    ''')
    db.commit()
    db.close()


dataset_dir = r"D:\DEMO\DeepDanbooru\what\grabber"
output_dir = r"D:\DEMO\DeepDanbooru\what\dataset"
make_dataset(dataset_dir,output_dir)
#
# dataset_dir_synthesis = r"D:\pycharmWorkspace\flaskProj\sample-syn-data"
# img_path = r"D:\stable-diffusion-webui\sd-scripts\models\rkrk12\rkrk12\examples\63626-3453064932-1girl, solo, cute girl,soles,toes,barefoot, feet,(spread legs_1.1),feet apart,off shoulder, short skirt,navel,underwear,masterpi.png"
# img = Image.open(img_path)
# print(img.info['parameters'])
# output_dir_synthesis = r"D:\pycharmWorkspace\flaskProj\sample-syn-data\dataset"
# make_dataset_synthesis(dataset_dir_synthesis,output_dir_synthesis)