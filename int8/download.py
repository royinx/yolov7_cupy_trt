# git clone https://github.com/waleedka/coco
# pip3 install cython matplotlib requests tqdm
# cd coco/PythonAPI && make && make install && python3 setup.py install && cd - 
# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
from pycocotools.coco import COCO
import requests
from tqdm import tqdm
import os
import threading
from queue import Queue

def load_categories():
    coco = COCO('annotations/instances_train2017.json')
    cats = coco.loadCats(coco.getCatIds())
    nms=[cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(nms)))


    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds )
    images = coco.loadImgs(imgIds)
    # print("imgIds: ", imgIds)
    # print("images: ", images)
    print("Finish Loading! \n")
    return images

def download(q, dst_dir, total):
    thread_id = threading.get_ident()
    print(f"Thread {thread_id} started")
    while q.empty() is False:
        if q.qsize() % 100 == 0:
            print(f"Remaining {q.qsize()} / {total}")
        im = q.get()
        img_data = requests.get(im['coco_url']).content
        with open(f'{dst}/' + im['file_name'], 'wb') as handler:
            handler.write(img_data)
        q.task_done()

if __name__ == "__main__":
    q = Queue()
    num_threads = 20
    dst = "downloaded_images"
    if not os.path.exists(dst): os.makedirs(dst)

    images = load_categories()
    for im in images:
        q.put(im)

    threads = [threading.Thread(target=download, args=(q,dst, len(images), )) for i in range(num_threads)]
    [thread.start() for thread in threads]
    [thread.join() for thread in threads]