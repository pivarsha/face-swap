# import cv2
# from deepface import DeepFace
# import numpy as np
# import os

# def extract_face(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"Failed to load image: {img_path}")
#         return None
    
#     try:
#         # Use extract_faces instead of detectFace
#         faces = DeepFace.extract_faces(img_path=img_path, detector_backend='opencv')
#         if not faces or len(faces) == 0:
#             print(f"No faces detected in {img_path}")
#             return None
        
#         # Extract the first detected face (assuming one face per image)
#         face = faces[0]['face']  # Returns normalized RGB image (0-1)
#         # Scale to 0-255 and convert to BGR
#         face = (face * 255).astype(np.uint8)
#         face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
#         return face
#     except Exception as e:
#         print(f"Face extraction failed: {str(e)}")
#         return None

# def swap_faces(source_path, target_path, output_path):
#     # Load images
#     source_img = cv2.imread(source_path)
#     target_img = cv2.imread(target_path)
    
#     if source_img is None or target_img is None:
#         print("Failed to load one or both images")
#         return
    
#     # Extract faces
#     source_face = extract_face(source_path)
#     target_face = extract_face(target_path)
    
#     if source_face is None or target_face is None:
#         print("Cannot proceed with face swap due to detection failure")
#         return
    
#     # Get target face coordinates
#     try:
#         target_detector = DeepFace.verify(target_path, target_path, enforce_detection=False)
#         target_coords = target_detector['facial_areas']['img1']
#     except Exception as e:
#         print(f"Failed to detect target coordinates: {str(e)}")
#         return
    
#     # Resize source face to match target
#     target_height = target_coords['h']
#     target_width = target_coords['w']
#     source_face_resized = cv2.resize(source_face, (target_width, target_height))
    
#     # Create a softer mask
#     mask = np.zeros_like(target_img, dtype=np.uint8)
#     mask[target_coords['y']:target_coords['y']+target_height, 
#          target_coords['x']:target_coords['x']+target_width] = 255
#     mask = cv2.GaussianBlur(mask, (21, 21), 0)  # Soften edges
    
#     # Basic lighting correction
#     target_roi = target_img[target_coords['y']:target_coords['y']+target_height, 
#                            target_coords['x']:target_coords['x']+target_width]
#     source_face_resized = cv2.addWeighted(source_face_resized, 0.95, target_roi, 0.05, 0.0)
    
#     # Seamless cloning
#     center = (target_coords['x'] + target_width//2, 
#               target_coords['y'] + target_height//2)
#     try:
#         output = cv2.seamlessClone(
#             source_face_resized,
#             target_img,
#             mask[target_coords['y']:target_coords['y']+target_height, 
#                  target_coords['x']:target_coords['x']+target_width],
#             center,
#             cv2.NORMAL_CLONE
#         )
        
#         # Save result
#         if cv2.imwrite(output_path, output):
#             print(f"Result saved to {output_path}")
#         else:
#             print(f"Failed to save output to {output_path}")
#     except Exception as e:
#         print(f"Seamless cloning failed: {str(e)}")

# def main():
#     source_path = "source.jpeg"
#     target_path = "target.jpeg"
#     output_path = "output.jpeg"
    
#     if not os.path.exists(source_path) or not os.path.exists(target_path):
#         print("Please ensure source.jpeg and target.jpeg exist")
#         return
    
#     swap_faces(source_path, target_path, output_path)

# if __name__ == "__main__":
#     main()



import cv2
from deepface import DeepFace
import numpy as np
import os

def extract_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return None
    
    try:
        faces = DeepFace.extract_faces(img_path=img_path, detector_backend='opencv')
        if not faces or len(faces) == 0:
            print(f"No faces detected in {img_path}")
            return None
        
        face = faces[0]['face']
        face = (face * 255).astype(np.uint8)
        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
        return face
    except Exception as e:
        print(f"Face extraction failed: {str(e)}")
        return None

def swap_faces(source_path, target_path, output_path, x_offset=0, y_offset=0, scale_factor=1.0):
    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)
    
    if source_img is None or target_img is None:
        print("Failed to load one or both images")
        return
    source_face = extract_face(source_path)
    target_face = extract_face(target_path)
    
    if source_face is None or target_face is None:
        print("Cannot proceed with face swap due to detection failure")
        return
    try:
        target_detector = DeepFace.verify(target_path, target_path, enforce_detection=False)
        target_coords = target_detector['facial_areas']['img1']
    except Exception as e:
        print(f"Failed to detect target coordinates: {str(e)}")
        return
    target_height = int(target_coords['h'] * scale_factor)
    target_width = int(target_coords['w'] * scale_factor)
    source_face_resized = cv2.resize(source_face, (target_width, target_height))
    center_x = target_coords['x'] + target_width // 2 + x_offset
    center_y = target_coords['y'] + target_height // 2 + y_offset
    center = (center_x, center_y)
    x_start = max(0, target_coords['x'] + x_offset)
    y_start = max(0, target_coords['y'] + y_offset)
    x_end = min(target_img.shape[1], x_start + target_width)
    y_end = min(target_img.shape[0], y_start + target_height)
    mask = np.zeros_like(target_img, dtype=np.uint8)
    mask[y_start:y_end, x_start:x_end] = 255
    mask = cv2.GaussianBlur(mask, (21, 21), 0)  # Soften edges
    target_roi = target_img[y_start:y_end, x_start:x_end]
    if target_roi.shape[:2] != source_face_resized.shape[:2]:
        source_face_resized = cv2.resize(source_face_resized, (target_roi.shape[1], target_roi.shape[0]))
    source_face_resized = cv2.addWeighted(source_face_resized, 0.95, target_roi, 0.05, 0.0)
    try:
        output = cv2.seamlessClone(
            source_face_resized,
            target_img,
            mask[y_start:y_end, x_start:x_end],
            center,
            cv2.NORMAL_CLONE
        )
        if cv2.imwrite(output_path, output):
            print(f"Result saved to {output_path} (x_offset={x_offset}, y_offset={y_offset}, scale_factor={scale_factor})")
        else:
            print(f"Failed to save output to {output_path}")
    except Exception as e:
        print(f"Seamless cloning failed: {str(e)}")

def main():
    source_path = "source.jpeg"
    target_path = "collage.jpeg"
    output_path = "output.jpeg"
    
    if not os.path.exists(source_path) or not os.path.exists(target_path):
        print("Please ensure source.jpeg and target.jpeg exist")
        return
    x_offset = 6    # Positive moves right, negative moves left
    y_offset = 9      # Positive moves down, negative moves up
    scale_factor = 1.1 # >1.0 enlarges, <1.0 shrinks
    
    swap_faces(source_path, target_path, output_path, x_offset, y_offset, scale_factor)

if __name__ == "__main__":
    main()