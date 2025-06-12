# -- Imports --
import whisper
import cv2
import os
import numpy as np
from moviepy import VideoFileClip, AudioFileClip

current_path=''#to be set by the main script

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
    path_to_video = os.path.join(current_path, video_name)
    video = VideoFileClip(path_to_video)
    return video

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
    audio = convert_video_to_audio(video_name)
    audio_path = os.path.join(current_path, os.path.splitext(video_name)[0])+"_audio.mp3"
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