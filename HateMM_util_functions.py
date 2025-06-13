# -- Imports --
import whisper
import cv2
import os
import pandas as pd
import numpy as np
from moviepy import VideoFileClip, AudioFileClip

video_folder_path=''#to be set by the main script
annotation_file_path = ''#to be set by the main script

def convert_video_to_audio(video_name: str) -> AudioFileClip:
    """
    Converts any valid video format in to mp3 audio to be
    fed in to OpenAIs Whisper model later.

    Params: video_name (str)
    Name of the video file.

    Returns: Audio file of type AudioFileClip.
    """
    video = get_video(video_name)
    audio = video.audio
    #TODO folderise each video such that they will each contain all of its processed info

    return audio

def get_video(video_name: str) -> VideoFileClip:
    """
    Retrieves path to the video and converts it to a
    video file to be processed programatically.

    Params: video_name (str)
    Name of the video file.

    Returns: Video file of type VideoFileClip.
    """
    path_to_video = os.path.join(video_folder_path, video_name)
    video = VideoFileClip(path_to_video)
    return video

def get_all_annotations(annotation_file_path: str) -> dict:
    """
    Retrieves the csv file of all annotations and converts
    them in to a list.

    Params: annotation_folder_path (str)
    Path to the aforementioned csv file.

    Returns: Dictionary of all annotations.
    """
    data = pd.read_csv(annotation_file_path)
    annotations_dict = data.to_dict(orient='records')
    return annotations_dict

def get_annotation(video_name: str, annotations: dict) -> str|None:
    """
    Queries the annotation of the specified video and
    returns it as string. May consider returning ALL
    labels instead of just the hate/non-hate label.

    Params: 
    video_name (str) - name of the video we are querying.
    annotations (dict) - all annotations of the dataset.

    Returns: Annotation as string or None if record doesn't exist.
    """
    record = next((row for row in annotations if row['video_file_name'] == video_name), None)
    if record:
        return str(record.get('label'))
    return None
#TODO
def get_video_frames(video):
    return 0

def get_audio_transcript(video_name: str) -> str|list:
    """
    Writes an audio file to the system and then uses
    the Whisper model to transcribe the audio in to
    textual format.

    Params: video_name
    Name of the video file.

    Returns: Transcribed audio in textual format of type string OR list.
    """
    #Retrieves audio from video, then writes audio file to users system
    audio = convert_video_to_audio(video_name)
    audio_path = os.path.join(video_folder_path, os.path.splitext(video_name)[0])+"_audio.mp3"
    audio.write_audiofile(audio_path)

    #Load model and extract audio transcription given the audio files' path
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    audio_transcript = result["text"]
    return audio_transcript

#TODO
def get_visual_description(video):
    return 0

#TODO
def export_to_folder(data):
    return 0