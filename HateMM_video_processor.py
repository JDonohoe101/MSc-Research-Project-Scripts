# -- Imports --
import os
import HateMM_util_functions as utils
from rich.progress import Progress, track
import time

#TODO: Instead of exporting to folders, export as PROMPTS

def process_videos(VIDEO_FOLDER_PATH: str, ANNOTATION_FILE_PATH: str) -> None:
    """
    This is the main function which extracts multimodal semantic features from
    videos in to prompts that can be fed in to both open and closed
    source LLMs.

    Params:
    VIDEO_FOLDER_PATH - Path to the video folder.
    ANNOTATION_FILE_PATH - Path to the annotation csv file.
    """    

    total_extracted_video_data_list = [] #list to hold all videos extracted data
    VALID_EXTENSIONS = ['.mp4', '.mov', '.avi', '.mkv'] #checking formats avoids trying to process hidden files

    #Counts number of videos in a folder to inform the progress bar total
    video_count=0 #counter variable
    for video in os.scandir(VIDEO_FOLDER_PATH):
            if video.is_file() and os.path.splitext(video.name)[1].lower() in VALID_EXTENSIONS:
                video_count += 1

    # -- Progress bar wrapper --
    with Progress() as progress:
        task = progress.add_task("[green]Processing videos...", total=video_count)
        all_annotations_dict = utils.get_all_annotations(ANNOTATION_FILE_PATH)

        # -- Main loop --
        for video in os.scandir(VIDEO_FOLDER_PATH):
            if video.is_file() and os.path.splitext(video.name)[1].lower() in VALID_EXTENSIONS:  # check if it's a (valid) video file
                video_name = video.name
                progress.console.log(f"Extracting data from {video_name}", style='bold magenta')

                # -- Calls to the extraction logic in util file --
                video_frames = utils.get_video_frames(video_name)
                video_audio_transcript = utils.get_audio_transcript(video_name)
                print(f"\nVIDEO: ", video_name)
                print(f"AUDIO TRANSCRIPT: ",video_audio_transcript,'\n')#debug statement, remove later
                video_visual_description = utils.get_visual_description(video_name)
                video_annotation = utils.get_annotation(video_name, all_annotations_dict)
                print(f"ANNOTATION: ", video_annotation,'\n')

                extracted_video_data = { #Dictionary used to hold all extracted data in a video
                    "video_name" : video_name,
                    "video_frames" : video_frames,
                    "video_audio_transcript" : video_audio_transcript,
                    "video_visual_description" : video_visual_description,
                    "video_annotation" : video_annotation
                }

                total_extracted_video_data_list.append(extracted_video_data)
                progress.update(task, advance=1) #advances progress bar by one step
                #time.sleep(0.01)  #debug statement for progress bar

        utils.export_to_folder(total_extracted_video_data_list) #call to export logic

# -- Initial logic --
if __name__ == "__main__":
    #Defines the path to the video folder and update the path in the utils module.
    VIDEO_FOLDER_PATH = "/Users/jackdonohoe/MSC RESEARCH PROJECT/Datasets/HateMM - Data/7799469/func_test"
    ANNOTATION_FILE_PATH = "/Users/jackdonohoe/MSC RESEARCH PROJECT/Datasets/HateMM - Data/7799469/HateMM_annotation.csv"
    utils.video_folder_path=VIDEO_FOLDER_PATH
    utils.annotation_file_path=ANNOTATION_FILE_PATH
    process_videos(VIDEO_FOLDER_PATH, ANNOTATION_FILE_PATH)