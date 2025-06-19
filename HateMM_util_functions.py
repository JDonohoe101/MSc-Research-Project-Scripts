# -- Imports --
import whisper
import cv2
import os
import pandas as pd
import numpy as np
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from moviepy import VideoFileClip, AudioFileClip

video_folder_path=''#to be set by the main script
annotation_file_path = ''#to be set by the main script

# -- Image caption model initialisation --
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

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

def get_video_frames(video_name: str):
    """
    Creates a directory for the extracted frames, samples the frames
    from the video provided, saves them in to the folder created.

    Params: video_name
    Name of the video we are processing.
    """
    # Load the video
    path_to_video = os.path.join(video_folder_path, video_name)
    cap = cv2.VideoCapture(path_to_video)
    base_name = os.path.splitext(video_name)[0]  # removes extension
    frame_folder = os.path.join(video_folder_path, f"{base_name}_frames")
    os.makedirs(frame_folder, exist_ok=True)

    # Check if video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_count = 0  # Initialize frame counter
    SAMPLE_RATE = 24 # how many frames before sampling next
    # Loop through each frame in the video
    while True:
        success, frame = cap.read()

        # Break the loop if the video ends
        if not success:
            break

        if frame_count % SAMPLE_RATE == 0:
            # Save the frame as an image
            frame_filename = os.path.join(frame_folder,f"{base_name}_frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            #print(f"Frame {frame_count} saved as {frame_filename}")#debug
        
        frame_count += 1

    # Release the video capture object
    cap.release()

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
    audio.write_audiofile(audio_path, logger=None)

    #Load model and extract audio transcription given the audio files' path
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    audio_transcript = result["text"]
    return audio_transcript

#TODO put this in the other py file

def generate_caption(processor: Blip2Processor, model: Blip2ForConditionalGeneration, image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption
    
def output_captions(video_name: str):
    output_captions = {}
    frame_folder = os.path.join(video_folder_path, f"{os.path.splitext(video_name)[0]}_frames")

    for filename in os.scandir(frame_folder):
        frame_path = os.path.join(frame_folder, filename)
        caption = generate_caption(processor, model, frame_path)
        print(f"{filename.name} CAPTION: {caption}")
        output_captions[filename] = caption
    return output_captions

#TODO
def export_to_folder(data):
    return 0