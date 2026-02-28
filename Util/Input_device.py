from pygame import joystick, key
from random import uniform, choice
from Util.Common_functions import RoundSign
from ComputerVision.pose_worker import PoseWorker

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
    def __init__(self, game, team=1, index=0, mode="none", pose_worker=None):
        self.game = game
        self.pose_worker = pose_worker
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

    def get_press(self, raw_input, extra_inputs=None, extra_commands=None):
        if extra_inputs is None:
            extra_inputs = []
        if extra_commands is None:
            extra_commands = []
        macro_press = bool(extra_inputs) or bool(extra_commands)

        self.inter_press = 0
        self.current_input.clear()

        # 2 = down
        # 4 = back
        # 6 = forward
        # 1 = down‑back
        # 3 = down‑forward
        # 5 = neutral
        # 8 = up

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
            if (raw_input[ind] == 1 and self.last_input[ind] == 0)
        ]
        released_buttons = [
            "r_b" + str(ind)
            for ind in range(1, len(raw_input))
            if (raw_input[ind] == 0 and self.last_input[ind] == 1)
        ]
        holded_buttons = [
            "h_b" + str(ind)
            for ind in range(1, len(raw_input))
            if (raw_input[ind] == 1 and self.last_input[ind] == 1)
        ]
        
        commands = []
        if extra_commands:
            commands.extend(extra_commands)
        
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
            + extra_inputs
            + pressed_buttons
            + released_buttons
            + holded_buttons
            + commands
        )

        if macro_press:
            self.inter_press = 1
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


dummy_input = InputDevice(None, 0, 0, "none")
