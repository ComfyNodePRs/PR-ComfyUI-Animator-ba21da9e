import cv2
import numpy as np
import torch
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet50

class Animator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "driving_video": ("IMAGE",),  # Now expects a batch of frames
                "lighting_style": (["Rembrandt Lighting", "Natural Light", "Front Light", "Backlight", "Soft Light", 
                                    "Hard Light", "Rim Light", "Loop Lighting", "Broad Lighting", "Short Lighting", 
                                    "Butterfly Lighting", "Split Lighting"],),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE")  # Now returns batches of frames
    FUNCTION = "animate_and_light"
    CATEGORY = "animation"

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.pose = self.mp_pose.Pose(static_image_mode=False)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        self.segmentation_model = deeplabv3_resnet50(pretrained=True).eval()

    def detect_and_rig_body(self, image):
        np_image = np.array(image)
        results = self.pose.process(cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
        if results.pose_landmarks:
            rig = {
                'joints': [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark],
                'connections': self.mp_pose.POSE_CONNECTIONS
            }
            return rig
        else:
            raise ValueError("No body detected in the image")

    def apply_facial_rigging(self, image):
        np_image = np.array(image)
        results = self.face_mesh.process(cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR))
        if results.multi_face_landmarks:
            face_rig = {
                'landmarks': [(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark]
            }
            return face_rig
        else:
            raise ValueError("No face detected in the image")

    def animate(self, source_image, driving_frames):
        animated_frames = []
        source_body = self.detect_and_rig_body(source_image)
        source_face = self.apply_facial_rigging(source_image)

        for frame in driving_frames:
            driving_body = self.detect_and_rig_body(frame)
            driving_face = self.apply_facial_rigging(frame)
            animated_frame = self.apply_motion(source_image, source_body, source_face, driving_body, driving_face)
            animated_frames.append(animated_frame)
        return animated_frames

    def apply_lighting(self, animated_output, lighting_style):
        lighting_functions = {
            "Rembrandt Lighting": self.apply_rembrandt_lighting,
            "Natural Light": self.apply_natural_lighting,
            "Front Light": self.apply_front_lighting,
            "Backlight": self.apply_backlighting,
            "Soft Light": self.apply_soft_lighting,
            "Hard Light": self.apply_hard_lighting,
            "Rim Light": self.apply_rim_lighting,
            "Loop Lighting": self.apply_loop_lighting,
            "Broad Lighting": self.apply_broad_lighting,
            "Short Lighting": self.apply_short_lighting,
            "Butterfly Lighting": self.apply_butterfly_lighting,
            "Split Lighting": self.apply_split_lighting
        }
        lighting_function = lighting_functions.get(lighting_style, lambda x: x)
        return [lighting_function(frame) for frame in animated_output]

    def create_greenscreen(self, lit_output):
        greenscreen_frames = []
        for frame in lit_output:
            input_tensor = transforms.ToTensor()(frame).unsqueeze(0)
            with torch.no_grad():
                output = self.segmentation_model(input_tensor)['out'][0]
            output_predictions = output.argmax(0)
            mask = output_predictions.byte().cpu().numpy()
            mask = (mask == 15).astype(np.uint8) * 255  # 15 is the label for person in COCO dataset
            green_bg = np.zeros(frame.shape, dtype=np.uint8)
            green_bg[:, :] = [0, 255, 0]  # Green color
            greenscreen_frame = np.where(mask[:, :, None] == 255, frame, green_bg)
            greenscreen_frames.append(Image.fromarray(greenscreen_frame))
        return greenscreen_frames

    def animate_and_light(self, image, driving_video, lighting_style):
        source_image = transforms.ToPILImage()(image.squeeze(0))
        driving_frames = [transforms.ToPILImage()(frame) for frame in driving_video]
        
        animated_output = self.animate(source_image, driving_frames)
        lit_output = self.apply_lighting(animated_output, lighting_style)
        greenscreen_output = self.create_greenscreen(lit_output)
        
        lit_output_tensor = torch.stack([transforms.ToTensor()(frame) for frame in lit_output])
        greenscreen_output_tensor = torch.stack([transforms.ToTensor()(frame) for frame in greenscreen_output])
        
        return (lit_output_tensor, greenscreen_output_tensor)

    #=============================================================================================#
    #
    #=============================================================================================#
    # Helper methods
    def apply_motion(self, source_image, source_body, source_face, driving_body, driving_face):
        # Convert PIL Image to numpy array
        source_np = np.array(source_image)
        
        # Create meshgrid for source image
        h, w = source_np.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Get source and driving keypoints
        source_kp = np.array(source_body['joints'] + source_face['landmarks'])
        driving_kp = np.array(driving_body['joints'] + driving_face['landmarks'])
        
        # Calculate the difference between source and driving keypoints
        diff = driving_kp - source_kp
        
        # Interpolate the difference to create a dense motion field
        motion_field_x = griddata(source_kp, diff[:, 0], (x, y), method='linear', fill_value=0)
        motion_field_y = griddata(source_kp, diff[:, 1], (x, y), method='linear', fill_value=0)
        
        # Apply the motion field to the source image
        x_new = x + motion_field_x
        y_new = y + motion_field_y
        
        # Warp the source image
        animated_frame = cv2.remap(source_np, x_new.astype(np.float32), y_new.astype(np.float32), 
                                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return Image.fromarray(animated_frame)

    def apply_rembrandt_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Create a gradient mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.ellipse(mask, (w//4, h//2), (w//4, h//2), 0, 0, 360, (1), -1)
        mask = cv2.GaussianBlur(mask, (w//10*2+1, h//10*2+1), 0)
        
        # Apply the mask to increase brightness on one side
        img = img.astype(np.float32)
        img[:,:,0] = img[:,:,0] * (1 + mask * 0.5)
        img[:,:,1] = img[:,:,1] * (1 + mask * 0.5)
        img[:,:,2] = img[:,:,2] * (1 + mask * 0.5)
        
        # Clip values and convert back to uint8
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_natural_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Slightly increase overall brightness and contrast
        img = img.astype(np.float32)
        img = img * 1.1
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Apply subtle vignette effect
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.circle(mask, (w//2, h//2), int(np.sqrt(h*h+w*w)/2), (1), -1)
        mask = cv2.GaussianBlur(mask, (w//10*2+1, h//10*2+1), 0)
        img = img.astype(np.float32) * (0.9 + 0.1 * mask)[:,:,np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_front_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Increase overall brightness
        img = img.astype(np.float32)
        img = img * 1.2
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_backlighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Create a gradient mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.circle(mask, (w//2, h//2), int(np.sqrt(h*h+w*w)/2), (1), -1)
        mask = cv2.GaussianBlur(mask, (w//5*2+1, h//5*2+1), 0)
        
        # Apply the mask to increase brightness around the edges
        img = img.astype(np.float32)
        img = img * (1 + (1-mask) * 0.5)[:,:,np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_soft_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Apply Gaussian blur to soften the image
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Slightly increase brightness
        img = img.astype(np.float32)
        img = img * 1.1
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_hard_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Increase contrast
        img = img.astype(np.float32)
        img = (img - 128) * 1.2 + 128
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel)
        
        return Image.fromarray(img)

    def apply_rim_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Create a gradient mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.circle(mask, (w//2, h//2), int(np.sqrt(h*h+w*w)/2), (1), -1)
        mask = cv2.GaussianBlur(mask, (w//20*2+1, h//20*2+1), 0)
        
        # Apply the mask to increase brightness around the edges
        img = img.astype(np.float32)
        img = img * (1 + (1-mask) * 0.8)[:,:,np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_loop_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Create a gradient mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.ellipse(mask, (w//3, h//2), (w//4, h//2), 0, 0, 360, (1), -1)
        mask = cv2.GaussianBlur(mask, (w//10*2+1, h//10*2+1), 0)
        
        # Apply the mask to increase brightness on one side
        img = img.astype(np.float32)
        img = img * (1 + mask * 0.3)[:,:,np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_broad_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Create a gradient mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.ellipse(mask, (w//4, h//2), (w//3, h//2), 0, 0, 360, (1), -1)
        mask = cv2.GaussianBlur(mask, (w//10*2+1, h//10*2+1), 0)
        
        # Apply the mask to increase brightness on one side
        img = img.astype(np.float32)
        img = img * (1 + mask * 0.4)[:,:,np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_short_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Create a gradient mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.ellipse(mask, (w*3//4, h//2), (w//3, h//2), 0, 0, 360, (1), -1)
        mask = cv2.GaussianBlur(mask, (w//10*2+1, h//10*2+1), 0)
        
        # Apply the mask to increase brightness on one side
        img = img.astype(np.float32)
        img = img * (1 + mask * 0.4)[:,:,np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_butterfly_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Create a gradient mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask = cv2.ellipse(mask, (w//2, h//3), (w//4, h//4), 0, 0, 360, (1), -1)
        mask = cv2.GaussianBlur(mask, (w//20*2+1, h//20*2+1), 0)
        
        # Apply the mask to increase brightness from above
        img = img.astype(np.float32)
        img = img * (1 + mask * 0.5)[:,:,np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)

    def apply_split_lighting(self, frame):
        # Convert PIL Image to numpy array
        img = np.array(frame)
        
        # Create a gradient mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:, :w//2] = 1
        mask = cv2.GaussianBlur(mask, (w//20*2+1, h//20*2+1), 0)
        
        # Apply the mask to increase brightness on one half
        img = img.astype(np.float32)
        img = img * (1 + mask * 0.5)[:,:,np.newaxis]
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img)
