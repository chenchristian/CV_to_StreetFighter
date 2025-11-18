# CV to StreetFighter

The goal of this project is to use pose estimation models (e.g., ViT-Pose, MediaPipe, or OpenPose) to detect human movements and map them to Street Fighter character actions.
 
By capturing player poses through a webcam, the system recognizes actions such as:

👊 Punch

🦶 Kick

🕺 Movement (left/right)

These inputs are then fed into a custom PyGame environment that controls Street Fighter gameplay.

## Set Up and Installation
```bash
git clone https://github.com/chenchristian/StreetFighter-PoseController.git
cd StreetFighter-PoseController

conda create -n StreetFighter python=3.12
conda activate StreetFighter
```

## Requirements
```bash
pip install pygame PyOpenGL
```
## Sprites

This game uses character sprites from *Street Fighter III* and *SNK vs Capcom* for educational purposes only.

You can download the sprites from the following source:

- [Ryu Street Fighter III (Sprite Sets)](https://www.nowak.ca/zweifuss/all/02_Ryu.zip)
- [Ken Street Fighter III (Sprite Sets)](https://www.nowak.ca/zweifuss/all/11_Ken.zip)
- [Ingame effects Street Fighter III (Sprite Sets)](https://www.justnopoint.com/zweifuss/all/22_Ingame%20Effects.zip)
- [SNK vs Capcom - Haohmaru (The Spriters Resource)](https://www.spriters-resource.com/download/42408/)
- [SNK vs Capcom - Terry Bogard (The Spriters Resource)](https://www.spriters-resource.com/download/42433/)

Unzip the file and place the folder in the `Assets/images` folder before running the game.

## Basic Controls

| Action              | Input                    |
|---------------------|--------------------------|
| Move Left           | ←                        |
| Move Right          | →                        |
| Crouch              | ↓                        |
| Jump                | ↑                        |
| Light Punch (LP)    | A                        |
| Medium Punch (MP)   | S                        |
| Heavy Punch (HP)    | D                        |
| Light Kick (LK)     | Q                        |
| Medium Kick (MK)    | W                        |
| Heavy Kick (HK)     | E                        |
| Special Move (e.g., Hadouken) | ↓ ↘ → + Punch |

## How to Run

```bash
git clone https://github.com/chenchristian/CV_to_StreetFighter
cd StreetFighter
python main.py
