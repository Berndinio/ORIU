from os import walk
import threading
from threading import Thread
from PIL import Image, ImageDraw
import os
semaphore = threading.BoundedSemaphore(value=8)

def cropImage(args):
    im = Image.open(args)
    width, height = im.size
    box = [(75,75), (115, 115)]
    newPath = args.replace("lfwCropped","lfwCroppedRectangle")
    os.makedirs(os.path.dirname(newPath), exist_ok=True)
    draw = ImageDraw.Draw(im)
    draw.rectangle(box, fill="black")
    im.save(newPath)
    semaphore.release() # increments the counter

if __name__ == "__main__":
    f = []
    for (dirpath, dirnames, filenames) in walk("lfw"):
        for dir in dirnames:
            for (dirpath, dirnames, filenames) in walk("lfwCropped/"+dir):
                for file in filenames:
                    file = "lfwCropped/"+str(dir)+"/"+file
                    #crop
                    thread = Thread(target = cropImage, args = (file, ))
                    semaphore.acquire() # decrements the counter
                    thread.start()
        break
