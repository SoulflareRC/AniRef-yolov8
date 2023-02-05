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


def make_dataset(dataset_dir,output_dir):
    '''
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
                info[0]=int(info[0])
                info.append(tag_count_general)
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


dataset_dir = r"D:\pycharmWorkspace\DeepDanbooru\test\grabber"
output_dir = r"D:\pycharmWorkspace\DeepDanbooru\test\MyDataset"
make_dataset(dataset_dir,output_dir)