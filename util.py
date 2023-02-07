import re
import os
import requests
#from main import secure_filename
from werkzeug.utils import secure_filename

def valid_image_url(url):
    pattern = r"(https?:\/\/.*\.(?:png|jpg))"
    return bool(re.match(pattern, url))


def valid_youtube_video_url(url):
    pattern = r"^((?:https?:)?\/\/)?((?:www|m)\.)?((?:youtube\.com|youtu.be))(\/(?:[\w\-]+\?v=|embed\/|v\/)?)([\w\-]+)(\S+)?$"
    return bool(re.match(pattern, url))


def save_image_from_url(image_url, folder_destination):
    image = requests.get(image_url).content
    image_extension = image_url.split(".")[-1]
    
    # Save the image
    with open(os.path.join(folder_destination, 'fimt' + '.' + image_extension), "wb+") as f:
        f.write(image)
        filename = f.name
        return filename


def save_image_from_form(file_request, folder_destination):
    if "file1" not in file_request:
        return -1

    f = file_request['file1']
    print(f"\033[93mfilename: {f.filename} \033[0m")

    if f.filename == "":
        return -1

    if check_video_or_image_file(f.filename) == 1:
        # os.path.join(app.config['UPLOAD_FOLDER'])
        filename = secure_filename(f.filename)
        f.save(os.path.join(folder_destination, filename))
        return filename, "video"
    else:
        filename = secure_filename(f.filename)
        print(f"\033[93msecure filename: {f.filename} \033[0m")
        f.save(os.path.join(folder_destination, 'fimt' + '.' + filename.split('.')[-1]))
        return filename, "image"


def check_video_or_image_file(file_name):
    valid_extensions = ["mp4", "mkv", "avi"]

    if file_name.split(".")[-1] in valid_extensions:
        return 1  # if the file is a video file
    else:
        return 0  # if the file is an image file
def tryint(s):
    try:
        return int(s)
    except:
        return s


def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)