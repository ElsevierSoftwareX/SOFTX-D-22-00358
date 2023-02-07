from pytube import YouTube  # pip install pytube
from urllib import request  # pip install urllib3


def youtube_video_downloader(link, destination):
    """
        Download video from youtube.

        Params:
        link: youtube video link
        destination: path of the directory where the video will save
        file_name: Give a name to the downloaded video

        return: video file name
                or -1 (if any error or network connection is occurred)downloaded
    """

    try:
        print(f"YouTube link1: {link}")
        yt = YouTube(link)
        downloaded_video = yt.streams.filter(file_extension='mp4').order_by('resolution')[-1].download(destination)
        print(f"YouTube link2: {link}")
        if "/" in downloaded_video:
            return downloaded_video.split("/")[-1]
        return downloaded_video.split("\\")[-1]
    except Exception as e:
        print(e)
        return -1


def download_video_using_url(url, video_name):
    """
        Download video from url

        Params:
            url: Video url
            video_name: Give the file name of the video. Can also pass the path with video name where the video
                        have to save. e.g. D:\\videos\\downloaded.mp4

        returns: 1 (if operation is successful) or -1 (for failure case)
    """

    try:
        request.urlretrieve(url, video_name)
        return 1
    except Exception as e:
        print(e)
        return -1


if __name__ == "__main__":
    result = youtube_video_downloader(
        # link="https://youtu.be/xWOoBJUqlbI",
        link="https://www.youtube.com/watch?v=xWOoBJUqlbI",
        destination="C:\\Users\\sayan\\Documents\\IIT Patna\\Website\\Flask App\\uploaded"
    )
    print(result)

    video_name = download_video_using_url(
        url="http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        video_name="C:\\Users\\sayan\\Documents\\IIT Patna\\Website\\Flask App\\uploaded\\downloaded.mp4"
    )
    print(video_name)
