from os import walk
import threading
from threading import Thread
from PIL import Image
import os
semaphore = threading.BoundedSemaphore(value=8)

def cropImage(args):
    im = Image.open(args)
    width, height = im.size
    box = (30,30,width-30, height-30)
    newPath = args.replace("lfw","lfwCropped")
    os.makedirs(os.path.dirname(newPath), exist_ok=True)
    im.crop(box).save(newPath)
    semaphore.release() # increments the counter

if __name__ == "__main__":
    f = []
    for (dirpath, dirnames, filenames) in walk("lfw"):
        for dir in dirnames:
            for (dirpath, dirnames, filenames) in walk("lfw/"+dir):
                for file in filenames:
                    file = "lfw/"+str(dir)+"/"+file
                    #crop
                    thread = Thread(target = cropImage, args = (file, ))
                    semaphore.acquire() # decrements the counter
                    thread.start()
        break
