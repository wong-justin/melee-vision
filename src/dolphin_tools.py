'''launch dolphin, mainly for recording replays'''

from pathlib import Path
import subprocess
import json
from secrets import token_hex as rand_hex
from time import sleep
from os import rename as move_file
import os
import shutil
# import ubjson
import slippi
import video_tools

# dolphin.ini important settings:
# (no seekbar)
#
# MainWindowPosX = 535
# MainWindowPosY = 17
# MainWindowWidth = 600
# MainWindowHeight = 620
# __ didn't adjust these really
# RenderWindowXPos = 485
# RenderWindowYPos = 57
# RenderWindowWidth = 779
# RenderWindowHeight = 636
#
# => 584x480 (original 73:60 ratio)
#
# EmulationSpeed > 1.0 doesn't seem to work, so no gains there
#
# PauseMovie = True ? hopefully stops replay after n frames; EDIT - it doesn't
#
# make sure muted

# trick to record multiple replays faster: use multiple dolphin instances

def record_replay(replay_filepath, output_dir='../vid_out/', slowdown_factor=1.1):

    # paths
    replay_filepath = Path(replay_filepath)
    dolphin_folder_path = Path(r'../dolphin')
    executable_path = dolphin_folder_path / 'Slippi Dolphin.exe'
    iso_path = read_json('../settings.json')['iso_path']
    tmp_output_folder = Path('../tmp/dolphin_process_0')    # if multiple dolphins output to same folder, there would be default filename conflicts with "framedump0.avi"
    real_output_folder = Path(output_dir)

    DEFAULT_VIDEO_NAME = 'framedump0.avi'
    old_video_path = tmp_output_folder  / DEFAULT_VIDEO_NAME
    new_video_name = replay_filepath.stem + '.avi'
    new_video_path = real_output_folder / new_video_name

    # clear tmp (could have files if error on last run)
    shutil.rmtree(tmp_output_folder)
    os.mkdir(tmp_output_folder)

    # get how long it should take to finish playback
    game = slippi.Game(replay_filepath)
    num_frames = len(game.frames)
    FPS = 60
    duration_sec = num_frames / FPS

    # prep instructions for slippi dolphin to replay
    comm_filepath = generate_commfile(replay_filepath)

    # start dolphin, giving commfile (location) and other misc flags
    # for standard dolphin cmd line, see
    #  https://wiki.dolphin-emu.org/index.php?title=Help:Contents
    process = subprocess.Popen([
        str(executable_path),
        '-i', str(comm_filepath),
        '--hide-seekbar',
        '-e', str(iso_path),
        '-b',
        # '-l',
        # '-d',
        '--output-directory', str(tmp_output_folder),
    ])

    # I wish I could get a dolphin.on_start_playback and on_finish
    #   but this crude dircheck and sleep() will have to suffice

    # wait until video file is created, signaling start
    # print('waiting on dolphin start..')
    while not os.listdir(tmp_output_folder):
        pass
    # print('dolphin has started playing')

    # wait in realtime for replay to finish
    # 4:02 (+123 f) of game duration only made 3:41 of video with slowdown; guess my computer is not fast enough for realtime
    # 244 s made 221 s, so multiply by factor of 244/221 ~= 1.104
    sleep(slowdown_factor * duration_sec) # overshoot with extra time as buffer

    # close dolphin, which will finish recording
    process.terminate()

    # give it time to close so video file can be modified
    sleep(1)

    # abort if video is too short (ie didn't finish)
    if video_tools.num_frames(str(old_video_path)) < num_frames:
        raise Exception('Video did not finish recording; trying again with slowdown_factor + 0.1 ?')
        # return record_replay(replay_filepath, output_dir, slowdown_factor+0.1)
        return -1

    # prep new location
    if new_video_path.exists():
        os.remove(new_video_path)
    # move_file(old_video_path, new_video_path)

    # trim excess "waiting for game.." frames and move to final location
    video_tools.trim(0, num_frames, str(old_video_path), str(new_video_path))


    # alternative idea to implement:
    # use SlippstreamClient (libmelee)
    # read replay file and use libmelee to set up game
    #     (characters, costumes, stage, ports, [tags?])
    # listen for gameevent or menuevent in slippstream client or something
    #     to signal game start and game end
    #
    #     could I start and stop recording on demand?

def play_replay(replay_filepath):
    # mainly just for testing
    replay_filepath = Path(replay_filepath)
    dolphin_folder_path = Path(r'../dolphin')
    executable_path = dolphin_folder_path / 'Slippi Dolphin.exe'
    iso_path = read_json('../settings.json')['iso_path']

    comm_filepath = generate_commfile(replay_filepath)

    return subprocess.Popen([
        str(executable_path),
        '-i', str(comm_filepath),
        # '-m', str(replay_filepath),
        '--hide-seekbar',
        '-e', str(iso_path),
        '-b',
        # '-C', '<System>.Movie.DumpFrames=False'
        # '-C', 'Movie.DumpFrames=False'
        # '-l',
        # '-d',
        # '--output-directory', str(tmp_output_folder),
    ])

def generate_commfile(replay_filepath):
    # comm file is used to tell slippi-dolhpin how to play a replay
    # https://github.com/project-slippi/slippi-wiki/blob/master/COMM_SPEC.md
    # I feel these should be turned into command line flags for simplicity/consistency with slippi dolphin cmd line

    unique_id = rand_hex(3*4)   # n bytes
    comm_filepath = Path(f'../tmp/commfile_{unique_id}.json')
    write_json(comm_filepath, {
        # 'mode': 'normal',   # default normal, or mirror or spectate
        'replay': str(replay_filepath),
        'isRealTimeMode': True,       # seems to prevent slowdowns?
        'commandId': unique_id,
        'startFrame': -123,
        # 'endFrame': num_frames,
    })
    return comm_filepath

# def enhanced_slippi_dolphin_argparse():
#     # remove need for callers to create commfile (-i)
#     # add dolphin configurability
#
#     custom_cmds = {
#         '-r', 'replay filepath'
#         '--config', '[section].[key]=[value]'
#     }
#     args = argparse.args()
#
#     # if there are custom commands, do them
#     for cmd in args:
#         if cmd in custom_cmds:
#             perform(cmd)
#             args.remove(cmd)
#
#     # do the rest of normal slippi dolphin cmdline
#     return subprocess.Popen(args)

def write_json(fp, data):
    with open(fp, 'w') as file:
        json.dump(data, file)

def read_json(fp):
    with open(fp, 'r') as file:
        return json.load(file)

if __name__ == '__main__':
    # record_replay(r'../test/Game_20210408T225110.slp', slowdown_factor=1.25)
    play_replay(r'../test/Game_20210408T225110.slp')
