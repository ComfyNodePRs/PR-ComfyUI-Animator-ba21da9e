# ComfyUI-Animator

This custom node for ComfyUI provides full-body animation capabilities, including facial rigging, various lighting styles, and green screen output.

## Features

- Full body detection and rigging
- Facial rigging for detailed animation
- Animation based on input driving video
- Multiple lighting styles
- Green screen output option

## Installation

1. Clone this repository into your ComfyUI custom nodes folder:
   ```
   https://github.com/SEkINVR/ComfyUI-Animator.git
   ```
   
2. Install the required dependencies:
   ```
   cd ComfyUI-Animator./install.sh
   ```
3.## Usage

1. In ComfyUI, you'll find the "Full Body Animator" node in the "animation" category.
2. Connect an image input, a driving video input, and select a lighting style.
3. The node will output two videos: one with the animated and lit subject, and another with the subject on a green screen background.

## Updating

To update the node to the latest version, run:
```
./update.sh
```

## License

[Your chosen license]
4.requirements.txt:
```
torch>=1.7.0
numpy>=1.19.0
Pillow>=8.0.0
opencv-python>=4.5.0
mediapipe>=0.8.9
```
5.install.sh:
```
#!/bin/bash
pip install -r requirements.txt
```
6. update.sh:
```
#!/bin/bash
git pull
pip install -r requirements.txt
