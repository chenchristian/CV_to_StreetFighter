from pygame import joystick, key
from random import uniform, choice
from Util.Common_functions import RoundSign
from ComputerVision.pose_worker import PoseWorker
from Util.network_peer import NetworkPeer

keyboard_mapping = (
    (79,),
    (80,),
    (82,),
    (81,),
    (8,),
    (26,),
    (20,),
    (7,),
    (22,),
    (4,),
    (21, 92),
    (21,),
    (116,),
)

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
                "keyboard": self.keyboard_mode,
                "joystick": self.joystick_mode,
                "AI": self.AI_mode,
                "record": self.record_mode,
                "none": self.none_mode,
                "random": self.random_mode,
                "network": self.network_mode,  # Receive inputs from network
                "network_sender": self.network_sender_mode,  # Send local inputs to network
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

        self.sequence_commands = (
            {"command": "QCF", "sequence": ("2", "3", "6"), "press": True},
            {"command": "QCF", "sequence": ("2", "6"), "press": True},
            {"command": "QCB", "sequence": ("2", "1", "4"), "press": True},
            {"command": "QCB", "sequence": ("2", "4"), "press": True},
            {"command": "DP", "sequence": ("6", "3", "2", "3"), "press": True},
            {"command": "DP", "sequence": ("3", "2", "3"), "press": True},#
            {"command": "DP", "sequence": ("3", "2", "6"), "press": True},
            {"command": "DP", "sequence": ("6", "2", "6"), "press": True},
            {"command": "DP", "sequence": ("6", "2", "3"), "press": True},
            {"command": "DP", "sequence": ("3", "2", "1", "3"), "press": True},
            {"command": "Doble_tap_forward", "sequence": ("5", "6", "5", "6")},
        )

        self.sequence_index = [1 for move in self.sequence_commands]

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
        if self.network_peer and self.network_peer.is_connected():
            self.network_peer.send_input(raw_input, frame=self.game.emu_frame)
        
        self.get_press(raw_input)

        

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
        processed_input = [
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
        
        # If network_peer is available, also send inputs to network
        if self.network_peer and self.network_peer.is_connected():
            self.network_peer.send_input(processed_input, frame=self.game.emu_frame)
        
        self.get_press(processed_input)

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
        """Receive inputs from network peer"""
        if self.network_peer is None:
            return
        
        # Check if connected
        if not self.network_peer.is_connected():
            # Not connected yet, use neutral input (11 elements: dpad + 10 buttons)
            self.get_press([[0,0],0,0,0,0,0,0,0,0,0,0])
            return
        
        # Get latest input from network
        network_input = self.network_peer.get_latest_input()
        if network_input is not None:
            # Ensure input has correct format (11 elements: dpad + 10 buttons)
            if len(network_input) < 11:
                # Pad with zeros if needed
                padded_input = list(network_input)
                while len(padded_input) < 11:
                    padded_input.append(0)
                network_input = padded_input
            self.get_press(network_input)
        else:
            # No input received, use neutral (11 elements: dpad + 10 buttons)
            self.get_press([[0,0],0,0,0,0,0,0,0,0,0,0])

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
        if self.network_peer:
            if self.network_peer.is_connected():
                self.network_peer.send_input(raw_input, frame=self.game.emu_frame)
            # If not connected, input is still used locally (will be sent once connected)
        
        # Use the input locally
        self.get_press(raw_input)

    def get_press(self, raw_input):
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
        for index, move in enumerate(self.sequence_commands):
            if (
                dpad
                is self.sequence_commands[index]["sequence"][self.sequence_index[index]]
                and dpad_trasition[0]
                is self.sequence_commands[index]["sequence"][
                    self.sequence_index[index] - 1
                ]
            ):
                self.sequence_index[index] = self.sequence_index[index] + 1
                if self.sequence_index[index] >= len(
                    self.sequence_commands[index]["sequence"]
                ):
                    self.sequence_index[index] = 1  # if move.get("press", False) else 1
                    commands.append(move["command"])

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
            + pressed_buttons
            + released_buttons
            + holded_buttons
            + commands
        )

        if raw_input != self.last_input:
            self.inter_press = 1
            if len(self.press_list_showed) > 20:
                self.press_list_showed.pop(0)
            self.press_list_showed.append(list(self.current_input))
            self.last_input_timer = self.input_timer
            self.input_timer = 0
        self.last_input = raw_input
        self.input_timer += 1

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


dummy_input = InputDevice(None, 0, 0, "none")
