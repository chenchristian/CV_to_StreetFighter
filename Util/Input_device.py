from pygame import joystick, key
from random import uniform, choice
from Util.Common_functions import RoundSign
from ComputerVision.pose_worker import PoseWorker
from Util.network_peer import NetworkPeer

keyboard_mapping = (
    (79,),#right arrow 0
    (80,),#left arrow 1
    (82,),#up arrow 2
    (81,),#down arrow 3
    (8,),#e 4
    (26,),#w 5
    (20,),#q 6
    (7,),#d 7
    (22,),#s 8
    (4,),#a 9
    (21, 92),#r, kp_4
    (21,), #r
    (116,),#execute??
)
combo_trail_key_mapping = {
    # Default SDL scancodes: t,y,u,i,o,p,g,h,j,k,l,z,x
    "Hadouken": (23,),
    "Shoryuken": (28,),
    "Shinku Hadouken": (24,),
    "Jab String": (12,),
    "Target Combo": (18,),
    "Front Fierce": (19,),
    "Tatsumaki": (10,),
    "Joudan Sokutogeri": (11,),
    "Release": (13,),
    "Aerial combo": (14,),
    "Joudan Sokutogeri EX": (15,),
    "String": (29,),
    "Aerial Tatsumaki EX": (27,),
}

# Hadouken → t (23)
# Shoryuken → y (28)
# Shinku Hadouken → u (24)
# Grab → i (12)
# Target Combo → o (18)
# Front Fierce → p (19)
# Tatsumaki → g (10)
# Joudan Sokutogeri → h (11)
# Release → j (13)
# Aerial combo → k (14)
# Joudan Sokutogeri EX → l (15)
# String → z (29)
# Aerial Tatsumaki EX → x (27)

joystick_name_mapping = {
    "Nintendo Switch Pro Controller": (
        (("analog", 0), ("binary", 14), ("binary", 13, 1)),
        (("analog", 1), ("binary", 11, 1), ("binary", 12)),
        (("analog", 5),),
        (("binary", 0),),
        (("binary", 1),),
        (("binary", 10),),
        (("binary", 2),),
        (("binary", 3),),
        (
            ("binary", 9),
            ("analog", 4),
        ),
    ),
    "Xbox Controller": (
        (("analog", 0),),
        (("analog", 1),),
        (("binary", 4),),
        (("binary", 3),),
        (("binary", 2),),
        (("binary", 5),),
        (("binary", 1),),
        (("analog", 2),),
        (("analog", 3),),
    ),
}

class InputDevice:
    def __init__(self, game, team=1, index=0, mode="none", pose_worker=None, network_peer=None):
        self.game = game
        self.pose_worker = pose_worker
        self.network_peer = network_peer
        self.type = "input"
        self.team, self.key, self.mode = (
            team,
            keyboard_mapping,
            {
                "external": self.external_mode, #using our AI
                "external_2": self.external_mode_2,
                "keyboard": self.keyboard_mode,
                "joystick": self.joystick_mode,
                "AI": self.AI_mode,
                "record": self.record_mode,
                "none": self.none_mode,
                "random": self.random_mode,
                "network": self.network_mode,  # Receive inputs from network
                "network_sender": self.network_sender_mode,  # Send local inputs to network
                "relay": self.relay_mode,  # Send local inputs AND receive from network (for relay server)
                "random_network": self.random_network_mode,  # CPU mode: random inputs sent to network
            }[mode],
        )
        if mode == "joystick":
            self.controller = joystick.Joystick(index)
            self.con = joystick_name_mapping[self.controller.get_name()]
       
        self.press_list_showed = []
        (
            self.current_input,
            self.raw_input,
            self.current_input,
            self.last_input,
            self.press_charge,
            self.inter_press,
        ) = (
            [[0, 0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ["5"],
            [[0, 0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            0,
        )
        
        self.input_timer = 0
        self.record_timer = 0
        self.recorded_inputs = []
        self.last_input_timer = 0
        self.recorded_inputs_index = 0
        self.draw_shake = [0, 0, 0, 0, 0, 0]
        self.rand_timer = 0
        self.active_object = None

        # 2 = down
        # 4 = back
        # 6 = forward
        # 1 = down‑back
        # 3 = down‑forward
        # 5 = neutral
        # 8 = up

        self.sequence_commands = (
            {"command": "QCF", "sequence": ("2", "3", "6"), "press": True},
            {"command": "QCF", "sequence": ("2", "6"), "press": True},
            {"command": "QCB", "sequence": ("2", "1", "4"), "press": True},
            {"command": "QCB", "sequence": ("2", "4"), "press": True},
            {"command": "DP", "sequence": ("6", "3", "2", "3"), "press": True},
            {"command": "DP", "sequence": ("3", "2", "3"), "press": True},
            {"command": "DP", "sequence": ("3", "2", "6"), "press": True},
            {"command": "DP", "sequence": ("6", "2", "6"), "press": True},
            {"command": "DP", "sequence": ("6", "2", "3"), "press": True},
            {"command": "DP", "sequence": ("3", "2", "1", "3"), "press": True},
            {"command": "Doble_tap_forward", "sequence": ("5", "6", "5", "6")},
        )

        self.sequence_index = [1 for move in self.sequence_commands]

        self.combo_trail_key_mapping = combo_trail_key_mapping
        self.combo_trail_macros = {
            "Hadouken": [["QCF"], ["p_b6"]],
            "Shoryuken": [["DP"], ["p_b6"]],
            "Shinku Hadouken": [["QCF"], ["QCF"], ["p_b6"]],
            "Jab String": [["p_b6"], ["p_b6"], ["p_b3"]],
            "Target Combo": [["2", "p_b5"], ["2", "p_b4"]],
            "Front Fierce": [["6", "p_b4"], ["p_b5"]],
            "Tatsumaki": [["2", "p_b2"], ["QCB", "p_b2"]],
            "Joudan Sokutogeri": [["p_b4"], ["QCF", "p_b3"]],
            "Release": [["p_b4"], ["p_b4", "p_b1"], ["2", "p_b1"]],
            "Aerial combo": [["p_b1"], ["2", "p_b5"], ["DP", "p_b5"]],
            "Joudan Sokutogeri EX": [["2", "p_b2"], ["QCF", "p_b3", "p_b2"], ["QCB", "p_b1"]],
            "String": [["p_b4"], ["2", "p_b5"], ["2", "p_b4"], ["p_b4", "p_b1"], ["p_b4"], ["QCF", "p_b6", "p_b5"]],
            "Aerial Tatsumaki EX": [["p_b4"], ["2", "p_b2"], ["DP", "p_b4"], ["p_b4", "p_b1"], ["p_b5"], ["QCB", "p_b3", "p_b2"], ["QCF"], ["QCF"], ["p_b6"]],
        }
        self.last_combo_trail_state = {
            title: 0 for title in self.combo_trail_key_mapping
        }

    def axis_button(self, input):
        return (
            (RoundSign(round(self.controller.get_axis(input[1]))))
            if input[0] == "analog"
            else (self.controller.get_button(input[1]) * (-1 if len(input) > 2 else 1))
        )

   
    def external_mode(self):
        if self.pose_worker is None:
            return  
        raw_input = self.pose_worker.get_latest_game_input()
        
        # If network_peer is available, also send inputs to network
        # Send input for NEXT frame (lockstep synchronization)
        if self.network_peer and self.network_peer.is_connected():
            self.network_peer.send_input(raw_input, frame=self.game.emu_frame + 1)
        
        self.get_press(raw_input)

    def external_mode_2(self):
        # Fallback to keyboard if pose_worker is None or not started
        if self.pose_worker is None:
            self.keyboard_mode()
            return
        
        # Check if pose worker thread is running
        thread_running = (hasattr(self.pose_worker, '_thread') and 
                         self.pose_worker._thread and 
                         self.pose_worker._thread.is_alive())
        
        if not thread_running:
            # Thread not running - fallback to keyboard
            self.keyboard_mode()
            return
        
        try:
            keyboard = self.pose_worker.get_latest_model_output()
            # Check if we got a valid keyboard tuple (should be 512 elements)
            # Even if all zeros, that's valid (just means no movement detected)
            if keyboard is None or not isinstance(keyboard, (tuple, list)):
                # Invalid format - fallback to keyboard
                self.keyboard_mode()
                return
            
            # Pad or truncate to 512 elements if needed
            if len(keyboard) < 512:
                keyboard = tuple(list(keyboard) + [0] * (512 - len(keyboard)))
            elif len(keyboard) > 512:
                keyboard = tuple(keyboard[:512])
        except (AttributeError, TypeError, Exception) as e:
            # Pose worker not ready or doesn't have model output yet - fallback to keyboard
            self.keyboard_mode()
            return
        
        self.raw_input = [
            sum(keyboard[key] for key in self.key[0]),
            sum(keyboard[key] for key in self.key[1]),
            sum(keyboard[key] for key in self.key[2]),
            sum(keyboard[key] for key in self.key[3]),
            sum(keyboard[key] for key in self.key[4]),
            sum(keyboard[key] for key in self.key[5]),
            sum(keyboard[key] for key in self.key[6]),
            sum(keyboard[key] for key in self.key[7]),
            sum(keyboard[key] for key in self.key[8]),
            sum(keyboard[key] for key in self.key[9]),
            sum(keyboard[key] for key in self.key[10]),
        ]
        
        combo_trail_inputs = []
        for title, keys in self.combo_trail_key_mapping.items():
            is_down = sum(keyboard[key] for key in keys) > 0
            was_down = self.last_combo_trail_state.get(title, 0)
            if is_down and not was_down:
                steps = self.combo_trail_macros.get(title, [])
                combo_trail_inputs = [item for step in steps for item in step]
            self.last_combo_trail_state[title] = 1 if is_down else 0

        self.get_press(
            [
                [
                    self.raw_input[0] + self.raw_input[1] * -1, # left right
                    self.raw_input[2] + self.raw_input[3] * -1, # up down
                ],
                self.raw_input[4],
                self.raw_input[5],
                self.raw_input[6],
                self.raw_input[7],
                self.raw_input[8],
                self.raw_input[9],
                self.raw_input[10],
            ],
            extra_inputs=combo_trail_inputs,
        )

        
    def keyboard_mode(self):
        keyboard = tuple(key.get_pressed())
        self.raw_input = [
            sum(keyboard[key] for key in self.key[0]),
            sum(keyboard[key] for key in self.key[1]),
            sum(keyboard[key] for key in self.key[2]),
            sum(keyboard[key] for key in self.key[3]),
            sum(keyboard[key] for key in self.key[4]),
            sum(keyboard[key] for key in self.key[5]),
            sum(keyboard[key] for key in self.key[6]),
            sum(keyboard[key] for key in self.key[7]),
            sum(keyboard[key] for key in self.key[8]),
            sum(keyboard[key] for key in self.key[9]),
            sum(keyboard[key] for key in self.key[10]),
        ]
        
        combo_trail_inputs = []
        for title, keys in self.combo_trail_key_mapping.items():
            is_down = sum(keyboard[key] for key in keys) > 0
            was_down = self.last_combo_trail_state.get(title, 0)
            if is_down and not was_down:
                steps = self.combo_trail_macros.get(title, [])
                combo_trail_inputs = [item for step in steps for item in step]
            self.last_combo_trail_state[title] = 1 if is_down else 0

        processed_input = [
            [
                self.raw_input[0] + self.raw_input[1] * -1, # left right
                self.raw_input[2] + self.raw_input[3] * -1, # up down
            ],
            self.raw_input[4],
            self.raw_input[5],
            self.raw_input[6],
            self.raw_input[7],
            self.raw_input[8],
            self.raw_input[9],
            self.raw_input[10],
        ]
        
        # If network_peer is available, also send inputs to network
        # Send input for NEXT frame (lockstep synchronization)
        if self.network_peer and self.network_peer.is_connected():
            self.network_peer.send_input(processed_input, frame=self.game.emu_frame + 1)
        
        self.get_press(processed_input, extra_inputs=combo_trail_inputs)

    def joystick_mode(self):
        self.raw_input = [
            (
                sum((self.axis_button(key)) for key in self.con[0]),
                -sum(self.axis_button(key) for key in self.con[1]),
            ),
            sum(self.axis_button(key) for key in self.con[2]),
            sum(self.axis_button(key) for key in self.con[3]),
            sum(self.axis_button(key) for key in self.con[4]),
            sum(self.axis_button(key) for key in self.con[5]),
            sum(self.axis_button(key) for key in self.con[6]),
            sum(self.axis_button(key) for key in self.con[7]),
            sum(self.axis_button(key) for key in self.con[8]),
        ]
        self.get_press(self.raw_input)

    def AI_mode(self):
        self.get_press(
            [
                [
                    self.raw_input[0] + self.raw_input[1] * -1,
                    self.raw_input[2] + self.raw_input[3] * -1,
                ],
                self.raw_input[4],
                self.raw_input[5],
                self.raw_input[6],
                self.raw_input[7],
                self.raw_input[8],
                self.raw_input[9],
                self.raw_input[10],
            ]
        )

    def record_mode(self):
        pass
        # enemy = get_object_per_team(self.team)

        # if game.record_input:
        #     if enemy.inputdevice.inter_press:
        #         self.recorded_inputs[-1][1] = enemy.inputdevice.last_input_timer

        #         self.recorded_inputs += [[enemy.inputdevice.raw_input, 0]]

        # if game.reproduce_input:
        #     self.record_timer += 1
        #     self.raw_input = self.recorded_inputs[self.recorded_inputs_index][0]

        #     self.get_press([[self.raw_input[0]+self.raw_input[1]*-1, self.raw_input[2]+self.raw_input[3]*-1], self.raw_input[4],
        #                     self.raw_input[5], self.raw_input[6], self.raw_input[7], self.raw_input[8], self.raw_input[9], self.raw_input[10]])

        #     if self.record_timer >= self.recorded_inputs[self.recorded_inputs_index][1]:
        #         self.recorded_inputs_index += 1
        #         self.record_timer = 0
        #         if self.recorded_inputs_index >= len(self.recorded_inputs):
        #             self.recorded_inputs_index = 0
        # else:
        #     self.raw_input = [[0, 0], 0, 0, 0, 0, 0, 0, 0, 0, 0]
        #     self.get_press(self.raw_input)

    def none_mode(self):
        self.raw_input = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.get_press([[0, 0], 0, 0, 0, 0, 0, 0, 0])

    def random_mode(self):
        self.rand_timer -= 1
        if self.rand_timer <= 0:
            self.rand_timer = uniform(*(10, 60))
            self.raw_input = [choice([0, 1]) for _ in range(11)]

        self.get_press(
            [
                [
                    self.raw_input[0] + self.raw_input[1] * -1,
                    self.raw_input[2] + self.raw_input[3] * -1,
                ],
                self.raw_input[4],
                self.raw_input[5],
                self.raw_input[6],
                self.raw_input[7],
                self.raw_input[8],
                self.raw_input[9],
                self.raw_input[10],
            ]
        )

    def network_mode(self):
        """Receive inputs from network peer - uses frame-based synchronization"""
        if self.network_peer is None:
            return
        
        # Check if connected
        if not self.network_peer.is_connected():
            # Not connected yet, use neutral input (11 elements: dpad + 10 buttons)
            self.get_press([[0,0],0,0,0,0,0,0,0,0,0,0])
            return
        
        # For lockstep: get input for the NEXT frame (which will be processed in gameplay())
        # This ensures both players process the same frame with the same inputs
        next_frame = self.game.emu_frame + 1
        network_input = self.network_peer.get_input_for_frame(next_frame)
        
        # If input not available, use prediction (last known input) instead of latest input
        # This prevents desync: using "latest" input could be from a future frame
        # Using "prediction" (last known) maintains consistency
        if network_input is None:
            network_input = self.network_peer.predict_input_for_frame(next_frame)
            
            # If still no input (no prediction available), use neutral input
            # This only happens at the very start before any inputs received
            if network_input is None:
                network_input = [[0,0],0,0,0,0,0,0,0,0,0,0]
        
        if network_input is not None:
            # Ensure input has correct format (11 elements: dpad + 10 buttons)
            if len(network_input) < 11:
                # Pad with zeros if needed
                padded_input = list(network_input)
                while len(padded_input) < 11:
                    padded_input.append(0)
                network_input = padded_input
            
            # CRITICAL: Reset double-tap sequence index before processing network input
            # This prevents false dash detection when receiving inputs over network
            # The sequence index can get out of sync between sender and receiver
            for idx, move in enumerate(self.sequence_commands):
                if move["command"] == "Doble_tap_forward":
                    # Only reset if we're partway through the sequence (not at start)
                    # This allows legitimate double-taps to still work
                    if self.sequence_index[idx] > 1:
                        # Check if this input would continue the sequence legitimately
                        expected_prev = move["sequence"][self.sequence_index[idx] - 1]
                        expected_curr = move["sequence"][self.sequence_index[idx]]
                        # Calculate what the transition would be
                        prev_dpad = [["8", "2", "5"], ["9", "3", "6"], ["7", "1", "4"]][self.last_input[0][0] * (1 if self.active_object == None else self.active_object.face)][self.last_input[0][1] - 1]
                        curr_dpad = [["8", "2", "5"], ["9", "3", "6"], ["7", "1", "4"]][network_input[0][0] * (1 if self.active_object == None else self.active_object.face)][network_input[0][1] - 1]
                        transition = prev_dpad + curr_dpad
                        # Only reset if transition doesn't match expected
                        if transition[0] != expected_prev or curr_dpad != expected_curr:
                            self.sequence_index[idx] = 1
            
            # Apply the input - get_press will handle last_input update correctly
            self.get_press(network_input)
        else:
            # No input received - DON'T use neutral as fallback to prevent false transitions
            # Instead, reuse last_input to maintain state and prevent false sequence detection
            # Only use neutral if we don't have a last_input yet
            if not hasattr(self, 'last_input') or self.last_input == [[0,0],0,0,0,0,0,0,0,0,0,0]:
                self.get_press([[0,0],0,0,0,0,0,0,0,0,0,0])
            # Otherwise, don't call get_press - this prevents false transitions
            # The character will continue with the last known input state

    def network_sender_mode(self):
        """Send local inputs to network and use them locally"""
        # Get local input from pose worker or keyboard
        if self.pose_worker:
            raw_input = self.pose_worker.get_latest_game_input()
        else:
            # Fallback to keyboard if no pose worker
            keyboard = tuple(key.get_pressed())
            self.raw_input = [
                sum(keyboard[key] for key in self.key[0]),
                sum(keyboard[key] for key in self.key[1]),
                sum(keyboard[key] for key in self.key[2]),
                sum(keyboard[key] for key in self.key[3]),
                sum(keyboard[key] for key in self.key[4]),
                sum(keyboard[key] for key in self.key[5]),
                sum(keyboard[key] for key in self.key[6]),
                sum(keyboard[key] for key in self.key[7]),
                sum(keyboard[key] for key in self.key[8]),
                sum(keyboard[key] for key in self.key[9]),
                sum(keyboard[key] for key in self.key[10]),
            ]
            raw_input = [
                [
                    self.raw_input[0] + self.raw_input[1] * -1,
                    self.raw_input[2] + self.raw_input[3] * -1,
                ],
                self.raw_input[4],
                self.raw_input[5],
                self.raw_input[6],
                self.raw_input[7],
                self.raw_input[8],
                self.raw_input[9],
                self.raw_input[10],
            ]
        
        # Send to network peer
        # Send input for NEXT frame (lockstep synchronization)
        if self.network_peer:
            if self.network_peer.is_connected():
                self.network_peer.send_input(raw_input, frame=self.game.emu_frame + 1)
            # If not connected, input is still used locally (will be sent once connected)
        
        # Use the input locally
        self.get_press(raw_input)
    
    def random_network_mode(self):
        """CPU mode: Generate random inputs and send them to network (for testing)"""
        # Generate random inputs (same as random_mode)
        self.rand_timer -= 1
        if self.rand_timer <= 0:
            self.rand_timer = uniform(*(10, 60))
            self.raw_input = [choice([0, 1]) for _ in range(11)]
        
        raw_input = [
            [
                self.raw_input[0] + self.raw_input[1] * -1,
                self.raw_input[2] + self.raw_input[3] * -1,
            ],
            self.raw_input[4],
            self.raw_input[5],
            self.raw_input[6],
            self.raw_input[7],
            self.raw_input[8],
            self.raw_input[9],
            self.raw_input[10],
        ]
        
        # Send to network peer
        if self.network_peer:
            if self.network_peer.is_connected():
                self.network_peer.send_input(raw_input, frame=self.game.emu_frame + 1)
        
        # Use the input locally
        self.get_press(raw_input)

    def relay_mode(self):
        """Send local inputs to network AND receive from network (for relay server mode).
        Always uses local input for this player's character, and sends it to network.
        Also receives network input (which is used by the other player's InputDevice)."""
        # Get local input from pose worker or keyboard
        raw_input = None
        
        if self.pose_worker:
            # Check if pose worker thread is running
            thread_running = (hasattr(self.pose_worker, '_thread') and 
                             self.pose_worker._thread and 
                             self.pose_worker._thread.is_alive())
            
            if thread_running:
                try:
                    # Use model output (same as external_mode_2) for camera input
                    keyboard = self.pose_worker.get_latest_model_output()
                    # Check if we got a valid keyboard tuple (should be 512 elements)
                    if keyboard is not None and isinstance(keyboard, (tuple, list)):
                        # Pad or truncate to 512 elements if needed
                        if len(keyboard) < 512:
                            keyboard = tuple(list(keyboard) + [0] * (512 - len(keyboard)))
                        elif len(keyboard) > 512:
                            keyboard = tuple(keyboard[:512])
                        
                        # Process keyboard tuple into raw_input format (same as external_mode_2)
                        self.raw_input = [
                            sum(keyboard[key] for key in self.key[0]),
                            sum(keyboard[key] for key in self.key[1]),
                            sum(keyboard[key] for key in self.key[2]),
                            sum(keyboard[key] for key in self.key[3]),
                            sum(keyboard[key] for key in self.key[4]),
                            sum(keyboard[key] for key in self.key[5]),
                            sum(keyboard[key] for key in self.key[6]),
                            sum(keyboard[key] for key in self.key[7]),
                            sum(keyboard[key] for key in self.key[8]),
                            sum(keyboard[key] for key in self.key[9]),
                            sum(keyboard[key] for key in self.key[10]),
                        ]
                        raw_input = [
                            [
                                self.raw_input[0] + self.raw_input[1] * -1,
                                self.raw_input[2] + self.raw_input[3] * -1,
                            ],
                            self.raw_input[4],
                            self.raw_input[5],
                            self.raw_input[6],
                            self.raw_input[7],
                            self.raw_input[8],
                            self.raw_input[9],
                            self.raw_input[10],
                        ]
                except (AttributeError, TypeError, Exception) as e:
                    # Error getting model output - fallback to keyboard
                    raw_input = None
        
        # Fallback to keyboard if no pose worker or if input format is wrong
        if raw_input is None:
            keyboard = tuple(key.get_pressed())
            self.raw_input = [
                sum(keyboard[key] for key in self.key[0]),
                sum(keyboard[key] for key in self.key[1]),
                sum(keyboard[key] for key in self.key[2]),
                sum(keyboard[key] for key in self.key[3]),
                sum(keyboard[key] for key in self.key[4]),
                sum(keyboard[key] for key in self.key[5]),
                sum(keyboard[key] for key in self.key[6]),
                sum(keyboard[key] for key in self.key[7]),
                sum(keyboard[key] for key in self.key[8]),
                sum(keyboard[key] for key in self.key[9]),
                sum(keyboard[key] for key in self.key[10]),
            ]
            raw_input = [
                [
                    self.raw_input[0] + self.raw_input[1] * -1,
                    self.raw_input[2] + self.raw_input[3] * -1,
                ],
                self.raw_input[4],
                self.raw_input[5],
                self.raw_input[6],
                self.raw_input[7],
                self.raw_input[8],
                self.raw_input[9],
                self.raw_input[10],
            ]
        
        # Always send local input to network (relay server will route it to other player)
        # Send input for NEXT frame (lockstep synchronization)
        if self.network_peer and self.network_peer.is_connected():
            next_frame = self.game.emu_frame + 1
            self.network_peer.send_input(raw_input, frame=next_frame)
            # Debug: print first few sends
            if next_frame < 10:
                print(f"[InputDevice] Player {self.team} sending input for frame {next_frame}: {raw_input[:2] if isinstance(raw_input, list) and len(raw_input) > 0 else raw_input}")
        
        # Always use local input for this player's character
        self.get_press(raw_input)
        
        # Note: The received network input (from other player) is handled by the other player's InputDevice
        # which is in "relay" mode and uses its own local input

    def get_press(self, raw_input, extra_inputs=None, extra_commands=None):
        if extra_inputs is None:
            extra_inputs = []
        if extra_commands is None:
            extra_commands = []
        macro_press = bool(extra_inputs) or bool(extra_commands)
        
        self.inter_press = 0
        self.current_input.clear()

        # Ensure last_input matches raw_input format (pad if needed)
        if len(self.last_input) != len(raw_input):
            # Pad last_input to match raw_input length
            if len(self.last_input) < len(raw_input):
                # Pad with zeros
                padded_last = list(self.last_input)
                while len(padded_last) < len(raw_input):
                    padded_last.append(0)
                self.last_input = padded_last
            else:
                # Truncate last_input to match raw_input
                self.last_input = list(self.last_input[:len(raw_input)])
                # Ensure first element is a list
                if len(self.last_input) > 0 and not isinstance(self.last_input[0], list):
                    self.last_input = [[0, 0]] + self.last_input[1:]

        # ↙↓↘←•→↖↑↗
        dpad = [["8", "2", "5"], ["9", "3", "6"], ["7", "1", "4"]][raw_input[0][0] * (1 if self.active_object == None else self.active_object.face)][
            raw_input[0][1] - 1
        ]
        dpad_trasition = str(
            [["8", "2", "5"], ["9", "3", "6"], ["7", "1", "4"]][self.last_input[0][0] * (1 if self.active_object == None else self.active_object.face)][
                self.last_input[0][1] - 1
            ]
        ) + str(dpad)


        pressed_buttons = [
            "p_b" + str(ind)
            for ind in range(1, len(raw_input))
            if (ind < len(raw_input) and ind < len(self.last_input) and raw_input[ind] == 1 and self.last_input[ind] == 0)
        ]
        released_buttons = [
            "r_b" + str(ind)
            for ind in range(1, len(raw_input))
            if (ind < len(raw_input) and ind < len(self.last_input) and raw_input[ind] == 0 and self.last_input[ind] == 1)
        ]
        holded_buttons = [
            "h_b" + str(ind)
            for ind in range(1, len(raw_input))
            if (ind < len(raw_input) and ind < len(self.last_input) and raw_input[ind] == 1 and self.last_input[ind] == 1)
        ]
        
        commands = []
        if extra_commands:
            commands.extend(extra_commands)
        
        for index, move in enumerate(self.sequence_commands):
            # Special handling for double-tap sequences to prevent false detection
            # Double-tap requires: neutral→forward→neutral→forward
            is_double_tap = move["command"] == "Doble_tap_forward"
            
            # Get expected previous state and current state for this sequence position
            expected_prev = self.sequence_commands[index]["sequence"][self.sequence_index[index] - 1]
            expected_curr = self.sequence_commands[index]["sequence"][self.sequence_index[index]]
            
            # Check if both the transition and current state match
            transition_matches = dpad_trasition[0] == expected_prev
            current_matches = dpad == expected_curr
            
            if transition_matches and current_matches:
                # Valid transition in sequence
                self.sequence_index[index] = self.sequence_index[index] + 1
                if self.sequence_index[index] >= len(
                    self.sequence_commands[index]["sequence"]
                ):
                    self.sequence_index[index] = 1  # Reset after completing sequence
                    commands.append(move["command"])
            else:
                # Transition doesn't match - reset sequence
                # For double-tap, be more strict: reset if we're not at the very start
                if is_double_tap and self.sequence_index[index] > 1:
                    # Reset double-tap sequence if we're partway through and transition doesn't match
                    self.sequence_index[index] = 1
                elif self.sequence_index[index] == 1 and dpad == expected_curr:
                    # At start of sequence and current input matches - keep waiting
                    pass
                else:
                    # Reset to beginning of sequence
                    self.sequence_index[index] = 1

        for index in range(1, 7):
            self.press_charge[index] = (
                (self.press_charge[index] + 1)
                if "h_b" + str(index) in holded_buttons
                else (0)
            )
            if self.press_charge[index] == 1 and pressed_buttons:
                pressed_buttons.append("p_b" + str(index))
            if self.press_charge[index] > 40:
                holded_buttons.append("charge_b" + str(index))
        self.current_input = (
            [dpad, dpad_trasition]
            + extra_inputs
            + pressed_buttons
            + released_buttons
            + holded_buttons
            + commands
        )

        # Set inter_press if input changed (for menu navigation and gameplay)
        # Check if dpad changed or buttons were pressed/released
        # Compare current dpad with last dpad to detect changes
        last_dpad = "5"  # Default to neutral
        if len(self.last_input) > 0 and isinstance(self.last_input[0], list):
            try:
                last_dpad = [["8", "2", "5"], ["9", "3", "6"], ["7", "1", "4"]][self.last_input[0][0] * (1 if self.active_object == None else self.active_object.face)][self.last_input[0][1] - 1]
            except (IndexError, TypeError):
                last_dpad = "5"
        
        if (dpad != last_dpad or pressed_buttons or released_buttons or macro_press):
            self.inter_press = 1
        else:
            self.inter_press = 0
        
        if macro_press:
            if len(self.press_list_showed) > 20:
                self.press_list_showed.pop(0)
            self.press_list_showed.append(list(self.current_input))
            self.last_input_timer = self.input_timer
            self.input_timer = 0

        self.last_input = raw_input
        self.input_timer += 1
        # print(self.current_input)

    def update(self, *args):
        self.mode()

    def draw(self, screen, pos, *args):
        if self.mode!= "none":
            for index in range(len(self.press_list_showed)):
                turn = 0
                for input in self.press_list_showed[index]:
                    if "reencor/" + input in self.game.image_dict:
                        image_key, image_size = self.game.image_dict["reencor/" + input]
                    else:
                        continue
                    screen.draw_texture(
                        image_key,
                        (
                            pos[0] + (-600 if self.team == 1 else 575) + 25 * turn * (1 if self.team == 1 else -1),
                            pos[1] -300 + 25 * (index),
                            -10,
                        ),
                        [20, 20],
                    )
                    turn += 1

#change none to random 
dummy_input = InputDevice(None, 0, 0, "random")
