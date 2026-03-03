import os
import string
import json
import warnings

# Suppress protobuf deprecation warnings from MediaPipe
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

from pygame import (
    init,
    quit,
    event,
    mixer,
    time,
    display,
    font,
    joystick,
    JOYDEVICEADDED,
)
from pygame.locals import *

from Util.Game_Screens import *
from Util.Common_functions import (
    dummy_json,
    get_object_per_class,
    get_object_per_team,
)
from Util.OpenGL_Renderer import (
    set_mode_opengl,
    load_image_path,
    font_texture,
    Camera,
    Screen,
)
from Util.Input_device import InputDevice, dummy_input
from Util.Interface_objects import Message

#global variable to activate\ deactivate the computer vision mode
COMPUTER_VISION = True
MODEL_PATH = "Models/LSTM_v1/phase1LSTM_original.pth"
lABEL_ENCODER = "Models/LSTM_v1/label_encoder.pkl"


current_dir = os.path.dirname(os.path.realpath(__file__))


def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))


def get_dictionaries(current_dir):
    object_dict, image_dict, sound_dict = {}, {}, {}
    path = current_dir + "/Assets"

    def get_parent_key(filepath):
        parts = filepath.replace("\\", "/").split("/")
        if len(parts) >= 2:
            folder = parts[-2]
            name = os.path.splitext(parts[-1])[0]
            return f"{folder}/{name}"
        else:
            return os.path.splitext(parts[-1])[0]

    def load_from_path(key, ext, full_path):
        if ext in ["png", "jpg", "jpeg"]:
            try:
                image_dict[key] = load_image_path(full_path)
            except Exception as e:
                print(f"Image load failed for {key}: {e}")
        elif ext in ["wav", "ogg", "mp3"]:
            try:
                sound_dict[key] = mixer.Sound(full_path)
            except Exception as e:
                print(f"Sound load failed for {key}: {e}")
        elif ext in ["json", "xml"]:
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    ob_json = dummy_json | json.load(f)
                    for box in dummy_json["boxes"]:
                        ob_json["boxes"][box] = dummy_json["boxes"][box] | ob_json[
                            "boxes"
                        ].get(box, {})
                    object_dict[key] = ob_json
            except Exception as e:
                print(f"JSON load failed for {key}: {e}")

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for name in files:
                ext = name.lower().split(".")[-1]
                full_path = os.path.join(root, name)
                key = get_parent_key(os.path.relpath(full_path, path))
                load_from_path(key, ext, full_path)

    font_type = font.Font(current_dir + "/Util/unispace bd.ttf", 60)
    for i in list(string.ascii_letters + string.digits) + [
        "+",
        "-",
        " ",
        ":",
        "_",
        "/",
        "!",
        "?",
        ".",
        ",",
        ";",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
    ]:
        image_dict["font " + i] = font_texture(font_type, i, (200, 200, 200))

    return image_dict, sound_dict, object_dict


class GameObject:
    def __init__(self, pose_worker=None, use_keyboard=False):
        self.pose_worker = pose_worker
        self.num_players = None  # Will be set by PlayerSelectionScreen (1 or 2)
        self.network_mode = False
        self.is_host = False
        self.relay_mode = False  # If True, using relay server instead of direct P2P
        self.network_peer = None
        self.remote_ip = "127.0.0.1"
        self.port = 5555
        self.use_keyboard = use_keyboard  # If True, use keyboard instead of camera
        self.type = "game"
        self.cpu_player1 = False  # Set via --cpu-player1 command line flag
        self.cpu_player2 = False  # Set via --cpu-player2 command line flag

        mixer.pre_init(44100, -16, 1, 1024)
        init()
        mixer.set_num_channels(16)

        self.resolution = (640, 400)
        self.internal_resolution = (1280, 800)
        self.time = time.Clock()
        self.frame_rate = 60

        set_mode_opengl(self.resolution)
        self.camera = Camera(0.1)
        self.screen = Screen(self.internal_resolution)
        display.set_caption("REENCOR")

        self.image_dict, self.sound_dict, self.object_dict = get_dictionaries(
            current_dir
        )
        self.object_list = []

        self.emu_frame = 0
        self.hitstop = 0
        
        # Frame synchronization for network mode
        self.sync_frame = 0  # Synchronized frame counter (shared between players)
        self.frame_wait_timeout = 0.1  # Max time to wait for frame input (100ms)
        self.camera_focus_point = [0, 0, -400]
        self.superstop, self.camera_path, self.frame, self.pos, self.draw_shake = (
            0,
            {},
            [0, 0],
            [10, 0, 0],
            [0, 0, 0, 0, 0, 0],
        )

        self.show_boxes = False
        self.show_inputs = False

        self.active = True
        self.player_number = 2
        self.selected_characters = "ryu SF3", "ryu SF3"
        self.selected_stage = "trining stage"

        # Initialize with default keyboard input for menu navigation
        # Will be reconfigured after player selection
        self.input_device_list = [InputDevice(self, 1, 1, "keyboard")]
        self.dummy_input_device = dummy_input
        
        # Desync detection and state management (for network mode)
        self.desync_detector = None
        self.state_manager = None

        self.screen_sequence = [
            PlayerSelectionScreen,  # Start with player selection
        ]
        self.current_screen = None
        self.screen_parameters = []

        # Uncomment below to skip player selection and go straight to game (for testing)
        # self.screen_sequence, self.selected_characters, self.selected_stage = (
        #     [VersusScreen],
        #     ["SF3/Ryu", "SF3/Ken"],
        #     ["Reencor/Training"],
        # )

        self.record_input = False
        self.reproduce_input = False

        self.active_players = []
        self.active_stages = None

    def Input_device_available(self):
        """Configure input devices based on number of players selected"""
        self.input_device_list = []
        
        # If num_players not set yet, use default single player
        if self.num_players is None:
            self.num_players = 1
        
        if self.num_players == 1:
            # Single player: use pose worker if available, else keyboard
            if COMPUTER_VISION and self.pose_worker:
                self.input_device_list = [
                    InputDevice(self, 1, 1, "external_2", pose_worker=self.pose_worker)
                ]
            elif self.pose_worker:
                self.input_device_list = [
                    InputDevice(self, 1, 1, "external", pose_worker=self.pose_worker)
                ]
            else:
                keyboard_count = 1
                for i in range(keyboard_count):
                    self.input_device_list.append(InputDevice(self, 1, i, "keyboard"))
            
            # joysticks
            joystick_count = joystick.get_count()
            for i in range(joystick_count):
                self.input_device_list.append(InputDevice(self, 2, i, "joystick"))
        elif self.num_players == 2:
            # Two players: network mode
            if self.network_mode:
                if self.relay_mode:
                    # Relay mode: use player_id from relay server to determine setup
                    # If player_id is None, default to 1 (shouldn't happen if connection worked)
                    player_id = self.network_peer.player_id if self.network_peer and self.network_peer.player_id else 1
                    
                    if player_id == 1:
                        # Player 1 (server): use relay_mode for player 1 to send local input, receive network input for player 2
                        # CPU mode: if --cpu-player1, use random_network mode instead of local input
                        if self.cpu_player1:
                            player1_mode = "random_network"
                            print("[Network] Player 1 set to CPU mode (random inputs)")
                        else:
                            player1_mode = "relay"
                        
                        if self.use_keyboard and not self.cpu_player1:
                            self.input_device_list = [
                                InputDevice(self, 1, 1, player1_mode,
                                          network_peer=self.network_peer),
                                InputDevice(self, 2, 2, "network",
                                          network_peer=self.network_peer)
                            ]
                        elif self.pose_worker and not self.cpu_player1:
                            self.input_device_list = [
                                InputDevice(self, 1, 1, player1_mode,
                                          pose_worker=self.pose_worker,
                                          network_peer=self.network_peer),
                                InputDevice(self, 2, 2, "network",
                                          network_peer=self.network_peer)
                            ]
                        else:
                            self.input_device_list = [
                                InputDevice(self, 1, 1, player1_mode,
                                          network_peer=self.network_peer),
                                InputDevice(self, 2, 2, "network",
                                          network_peer=self.network_peer)
                            ]
                    else:
                        # Player 2 (client): receive network input for player 1, use relay_mode for player 2 to send local input
                        # CPU mode: if --cpu-player2, use random_network mode instead of local input
                        if self.cpu_player2:
                            player2_mode = "random_network"
                            print("[Network] Player 2 set to CPU mode (random inputs)")
                        else:
                            player2_mode = "relay"
                        
                        self.input_device_list = [
                            InputDevice(self, 1, 1, "network",
                                      network_peer=self.network_peer)
                        ]
                        if self.use_keyboard and not self.cpu_player2:
                            self.input_device_list.append(
                                InputDevice(self, 2, 2, player2_mode,
                                          network_peer=self.network_peer)
                            )
                        elif self.pose_worker and not self.cpu_player2:
                            self.input_device_list.append(
                                InputDevice(self, 2, 2, player2_mode,
                                          pose_worker=self.pose_worker,
                                          network_peer=self.network_peer)
                            )
                        else:
                            self.input_device_list.append(
                                InputDevice(self, 2, 2, player2_mode,
                                          network_peer=self.network_peer)
                            )
                elif self.is_host:
                    # Host: Player 1 uses local input, Player 2 receives from network
                    if self.use_keyboard:
                        # Use keyboard for Player 1
                        self.input_device_list = [
                            InputDevice(self, 1, 1, "keyboard",
                                      network_peer=self.network_peer),
                            InputDevice(self, 2, 2, "network",
                                      network_peer=self.network_peer)
                        ]
                    elif self.pose_worker:
                        # Use camera for Player 1
                        self.input_device_list = [
                            InputDevice(self, 1, 1, "external", 
                                      pose_worker=self.pose_worker,
                                      network_peer=self.network_peer),
                            InputDevice(self, 2, 2, "network",
                                      network_peer=self.network_peer)
                        ]
                    else:
                        # Fallback to keyboard
                        self.input_device_list = [
                            InputDevice(self, 1, 1, "keyboard",
                                      network_peer=self.network_peer),
                            InputDevice(self, 2, 2, "network",
                                      network_peer=self.network_peer)
                        ]
                else:
                    # Client: Player 1 receives from network, Player 2 uses local input
                    self.input_device_list = [
                        InputDevice(self, 1, 1, "network",
                                  network_peer=self.network_peer)
                    ]
                    # Use keyboard if flag is set, otherwise try camera
                    # IMPORTANT: Client's Player 2 must use network_sender mode to send inputs to server
                    if self.use_keyboard:
                        self.input_device_list.append(
                            InputDevice(self, 2, 2, "network_sender",
                                      network_peer=self.network_peer)
                        )
                    elif self.pose_worker:
                        self.input_device_list.append(
                            InputDevice(self, 2, 2, "network_sender",
                                      pose_worker=self.pose_worker,
                                      network_peer=self.network_peer)
                        )
                    else:
                        self.input_device_list.append(
                            InputDevice(self, 2, 2, "network_sender",
                                      network_peer=self.network_peer)
                        )
            else:
                # Network not initialized yet - use fallback
                keyboard_conut = 1
                joystick_count = joystick.get_count()
                self.input_device_list = [
                    InputDevice(self, 1, 1, "keyboard")
                ]
                if joystick_count > 0:
                    for i in range(joystick_count):
                        self.input_device_list.append(
                            InputDevice(self, 2, i, "joystick")
                        )
                else:
                    self.input_device_list.append(
                        InputDevice(self, 2, 2, "none")
                    )
>>>>>>> network-multiplayer

    def next_screen(self, screen_sequence: list = [TitleScreen]):
        self.active = False
        self.screen_sequence += screen_sequence

    def screen_manager(self):
        while len(self.screen_sequence):
            self.active = True
            self.current_screen = self.screen_sequence[-1](
                *[self] + self.screen_parameters
            )
            self.screen_parameters = []
            while self.active:
                for dev in self.input_device_list + [self.dummy_input_device]:
                    dev.update()

                self.camera.update(self.camera_focus_point)
                self.current_screen.__loop__()

                self.screen.display()

                # --- ADD POSE VIEWER POLLING HERE ---
                try:
                    if hasattr(self, 'pose_viewer') and self.pose_viewer is not None:
                        self.pose_viewer.poll()
                except:
                    pass
                # ------------------------------------

                display.flip()
                self.event_handler()
                self.time.tick(self.frame_rate)

            self.screen_sequence.pop()
            self.current_screen.__dein__()

            self.hitstop = 0
            self.camera_focus_point = [0, 0, 400]
            self.superstop, self.camera_path, self.frame, self.pos, self.draw_shake = (
                0,
                {},
                [0, 0],
                [10, 0, 0],
                [0, 0, 0, 0, 0, 0],
            )
            self.show_boxes = False
            self.active = True
            self.record_input = False
            self.reproduce_input = False
            self.active_players = []
            self.active_stage = None

    def event_handler(self):
        for individual_event in event.get():
            if individual_event.type == QUIT:
                self.active = False
                quit()
                exit()
            if individual_event.type == KEYDOWN:
                if individual_event.key == K_0:
                    self.active = False
                    exit()
                if individual_event.key == K_9:
                    self.active = False
                    self.screen_manager()
            if individual_event.type == JOYDEVICEADDED:
                i = individual_event.device_index
                self.input_device_list.append(InputDevice(self, 2, i, "joystick"))
                self.active_players[1].inputdevice = self.input_device_list[-1]
                self.input_device_list[-1].active_object = self.active_players[1]
                self.object_list.append(
                    Message(
                        game=self,
                        string="Joystick connected",
                        texture_string=[
                            {"image": "reencor/+", "size": (70, 70)},
                            {"image": "reencor/5", "size": (70, 70)},
                        ],
                        pos=[self.resolution[0] * 0.9, self.resolution[1] * 0.8, -1],
                        background=(0, 0, 0, 126),
                        time=60,
                        kill_on_time=True,
                        allign="left",
                        scale=[0.5, 0.5],
                    )
                )

    def calculate_camera_focus_point(self):
        xpos_list = [active_object.pos[0] for active_object in self.active_players]
        ypos_list = [active_object.pos[1] for active_object in self.active_players]
        self.pos = [
            (sum(xpos_list) / len(xpos_list)),
            (sum(ypos_list) / len(ypos_list)) + self.resolution[1] * 0.6,
            400,
        ]

        camera_limits = self.active_stages[0].dict["camera_focus_point_limit"]
        half_width = self.internal_resolution[0] * 0.5
        half_height = self.internal_resolution[1] * 0.5

        if self.pos[0] - half_width < camera_limits[0][0]:
            self.pos[0] = camera_limits[0][0] + half_width
        elif self.pos[0] + half_width > camera_limits[0][1]:
            self.pos[0] = camera_limits[0][1] - half_width

        if self.pos[1] + half_height > camera_limits[1][0]:
            self.pos[1] = camera_limits[1][0] - half_height
        elif self.pos[1] - half_height < camera_limits[1][1]:
            self.pos[1] = camera_limits[1][1] + half_height

        if self.camera_path:
            path_frame = self.camera_path["path"][self.frame[1]]
            obj = self.camera_path["object"]
            scale = abs(400 / path_frame["pos"][2])
            zoom = (
                abs(self.camera_focus_point[2] / 400)
                if hasattr(self, "camera_focus_point")
                else scale
            )
            cx = path_frame["pos"][0] * obj.face + obj.pos[0]
            cy = path_frame["pos"][1] + obj.pos[1]
            cz = round(scale)
            half_w = self.internal_resolution[0] * 0.5 * zoom
            half_h = self.internal_resolution[1] * 0.5 * zoom
            min_x = self.pos[0] - self.internal_resolution[0] * 0.5 + half_w
            max_x = self.pos[0] + self.internal_resolution[0] * 0.5 - half_w
            min_y = self.pos[1] - self.internal_resolution[1] * 0.5 + half_h
            max_y = self.pos[1] + self.internal_resolution[1] * 0.5 - half_h
            self.camera_focus_point = [
                clamp(cx, min_x, max_x),
                clamp(cy, min_y, max_y),
                cz,
            ]
        else:
            self.camera_focus_point = self.pos

        if self.camera_path:
            self.frame[0] += 1
            if self.frame[0] > self.camera_path["path"][self.frame[1]]["dur"]:
                self.frame = [0, self.frame[1] + 1]
                if self.frame[1] >= len(self.camera_path["path"]):
                    self.camera_path, self.frame = {}, [0, 0]

    def gameplay(self, *args):
        # Server-authoritative network mode
        if self.network_mode and self.network_peer and self.network_peer.is_connected():
            if self.network_peer.is_server:
                # SERVER (Player 1): Authoritative simulation
                # Server simulates the game and sends state to clients
                
                # InputDevice.update() is called before gameplay(), so inputs are already processed
                # Player 1 input: already processed by InputDevice[0] (relay_mode - sends and uses local input)
                # Player 2 input: already processed by InputDevice[1] (network_mode - gets input from network buffer)
                # No need to manually call get_press() - InputDevice[1].update() already did it via network_mode()
                
                # Simulate frame (authoritative)
                # InputDevice[0] already has Player 1's input processed
                self.emu_frame += 1
                for object in self.object_list:
                    object.update(self.camera_focus_point)
                self.hitstop = self.hitstop - 1 if self.hitstop else 0
                
                for object in self.object_list:
                    update_display_shake(object)
                calculate_boxes_collitions(self)
                update_display_shake(self.camera)
                self.calculate_camera_focus_point()
                
                # Send state to clients periodically
                self.network_peer.send_game_state(self, self.emu_frame)
                
            elif self.network_peer.is_client:
                # CLIENT (Player 2): Receive state from server
                # Client sends inputs but doesn't simulate authoritatively
                
                # IMPORTANT: InputDevice.update() is called before gameplay(), so inputs are already sent
                # The client's InputDevice[1] in relay_mode sends local input to server
                
                # Check for state update from server
                latest_state = self.network_peer.get_latest_state()
                if latest_state:
                    target_frame = latest_state.frame
                    
                    # Always apply server state - server is authoritative
                    # This ensures projectiles and all game objects are synchronized
                    if target_frame >= self.emu_frame:
                        # Server is at or ahead of client - apply server state
                        # This will set emu_frame to target_frame and restore all game state
                        latest_state.apply_to_game(self)
                        # State application sets emu_frame, so we're synchronized
                    # If target_frame < emu_frame, we're ahead (shouldn't happen in server-authoritative)
                    # But still apply to ensure we don't miss any updates
                    elif target_frame < self.emu_frame:
                        # Client is ahead - still apply state to catch up (shouldn't happen normally)
                        latest_state.apply_to_game(self)
                else:
                    # No state available yet - wait for server
                    # Don't simulate ahead of server to maintain synchronization
                    # But InputDevice.update() still runs and sends inputs
                    pass
        else:
            # Local mode (no network) or not connected - normal simulation
            self.emu_frame += 1
            for object in self.object_list:
                object.update(self.camera_focus_point)
            self.hitstop = self.hitstop - 1 if self.hitstop else 0

            for object in self.object_list:
                update_display_shake(object)
            calculate_boxes_collitions(self)
            update_display_shake(self.camera)
            self.calculate_camera_focus_point()

    def display(self, *args):
        for object in self.object_list:
            object.draw(self.screen, self.camera.pos)
            if self.show_boxes:
                draw_boxes(self, object)
        if self.show_inputs:
            for dev in self.input_device_list:
                dev.draw(self.screen, self.camera.pos)

import torch
import pickle
from ComputerVision.pose_worker import PoseWorker
from ComputerVision.pose_viewer import PoseViewer

#change this to change which model we are using
from Models.LSTM_v1.lstm_live_predictions import LivePosePredictor
from Models.LSTM_v1.lstm_model import LSTMWindowClassifier

# Load model
with open(lABEL_ENCODER, "rb") as f:
    label_encoder = pickle.load(f)

model = LSTMWindowClassifier(
    input_size=84,  # 21 landmarks * 4 features
    hidden_size=128,
    num_layers=2,
    num_classes=len(label_encoder.classes_),
    dropout=0.3
)
if(COMPUTER_VISION):
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))  # or "cuda"
    model.eval()

    # Create predictor
    # change this to change which model we are using
    predictor = LivePosePredictor(model, label_encoder, sequence_length=5)

    # Check for --no-camera or --keyboard flag
    import sys
    use_keyboard = "--no-camera" in sys.argv or "--keyboard" in sys.argv

    # Initialize pose worker (don't start it yet - will be started based on player selection)
    # Only create if not using keyboard
    pose_worker = None if use_keyboard else PoseWorker(camera_index=0, live_predictor=predictor)
    
    # Start PoseViewer if pose_worker exists
    pose_viewer = None
    if pose_worker:
        pose_viewer = PoseViewer(shared_state=pose_worker)

    # Start game (pose worker will be started based on selection)
    game = GameObject(pose_worker=pose_worker, use_keyboard=use_keyboard)
    game.screen_manager()
else:
    # Check for --no-camera or --keyboard flag
    import sys
    use_keyboard = "--no-camera" in sys.argv or "--keyboard" in sys.argv
    
    # Start game without pose worker
    game = GameObject(pose_worker=None, use_keyboard=use_keyboard)
    game.screen_manager()
