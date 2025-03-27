# import cv2
# import dlib
# import numpy as np

# # Load face detector and landmark predictor
# PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  # Path to dlib's landmark predictor
# # PREDICTOR_PATH = "/Users/bharatdigital/Documents/face swaping/shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(PREDICTOR_PATH)

# def get_landmarks(image):
#     """Detect face landmarks."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)
#     if len(faces) == 0:
#         return None
#     return np.array([[p.x, p.y] for p in predictor(image, faces[0]).parts()], dtype=np.int32)

# def triangulation_points(landmarks, size):
#     """Get triangles based on landmarks."""
#     rect = (0, 0, size[1], size[0])
#     subdiv = cv2.Subdiv2D(rect)
#     print("Landmarks for triangulation:", landmarks)
#     for p in landmarks:
#         if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 2:
#             subdiv.insert((float(p[0]), float(p[1])))
    
#     triangle_list = subdiv.getTriangleList()
    
#     if len(triangle_list) == 0:
#         raise ValueError("Delaunay triangulation failed. Check the input landmarks.")

#     return triangle_list.astype(np.int32)


# def warp_triangle(src, dst, src_tri, dst_tri, dst_img):
#     """Warp a triangular region from src to dst."""
#     src_rect = cv2.boundingRect(src_tri)
#     dst_rect = cv2.boundingRect(dst_tri)
    
#     src_crop = src[src_rect[1]:src_rect[1]+src_rect[3], src_rect[0]:src_rect[0]+src_rect[2]]
#     src_tri_adj = src_tri - np.array([src_rect[:2]])
#     dst_tri_adj = dst_tri - np.array([dst_rect[:2]])

#     warp_mat = cv2.getAffineTransform(np.float32(src_tri_adj), np.float32(dst_tri_adj))
#     dst_crop = cv2.warpAffine(src_crop, warp_mat, (dst_rect[2], dst_rect[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

#     mask = np.zeros((dst_rect[3], dst_rect[2]), dtype=np.uint8)
#     cv2.fillConvexPoly(mask, np.int32(dst_tri_adj), 255)

#     dst_crop = cv2.bitwise_and(dst_crop, dst_crop, mask=mask)

#     dst_img[dst_rect[1]:dst_rect[1]+dst_rect[3], dst_rect[0]:dst_rect[0]+dst_rect[2]] = cv2.add(dst_img[dst_rect[1]:dst_rect[1]+dst_rect[3], dst_rect[0]:dst_rect[0]+dst_rect[2]], dst_crop)

# def seamless_clone(face_src, face_dst, dst_img, landmarks_dst):
#     """Blend the swapped face seamlessly using Poisson blending."""
#     mask = np.zeros_like(dst_img, dtype=np.uint8)
#     hull = cv2.convexHull(landmarks_dst)

#     # Ensure mask is correctly set
#     cv2.fillConvexPoly(mask, hull, (255, 255, 255))

#     # Compute center correctly
#     center = tuple(np.mean(hull.squeeze(), axis=0).astype(int))

#     # Ensure center is a valid (x, y) tuple
#     if len(center) != 2:
#         print("Error: Center of the face is invalid:", center)
#         return None

#     return cv2.seamlessClone(face_src, dst_img, mask, tuple(center), cv2.NORMAL_CLONE)


# def swap_faces(image1, image2):
#     """Perform face swap between image1 and image2."""
#     landmarks1 = get_landmarks(image1)
#     landmarks2 = get_landmarks(image2)

#     if landmarks1 is None or landmarks2 is None:
#         print("No face detected in one of the images.")
#         return None

#     triangles = triangulation_points(landmarks1, image1.shape)

#     swapped_face = np.zeros_like(image2)

#     for t in triangles.reshape(-1, 6):
#         x1, y1, x2, y2, x3, y3 = t
#         tri1 = np.array([[x1, y1], [x2, y2], [x3, y3]], np.int32)
#         tri2 = landmarks2[np.where((landmarks1 == tri1[:, None]).all(-1))[1]]

#         if len(tri2) == 3:
#             warp_triangle(image1, image2, tri1, tri2, swapped_face)

#     return seamless_clone(swapped_face, image2, image2.copy(), landmarks2)

# if __name__ == "__main__":
#     img1 = cv2.imread("source.jpeg")  # Replace with your image path
#     img2 = cv2.imread("target1.jpeg")  # Replace with your image path

#     output = swap_faces(img1, img2)

#     if output is not None:
#         cv2.imshow("Face Swapped", output)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import dlib

# # Load face detector and landmark predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Function to extract facial landmarks
# def get_face_landmarks(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     faces = detector(gray)

#     if len(faces) == 0:
#         return None, None

#     face_landmarks = []
#     face_rects = []
#     for face in faces:
#         landmarks = predictor(gray, face)
#         points = np.array([(p.x, p.y) for p in landmarks.parts()], np.int32)
#         face_landmarks.append(points)
#         face_rects.append(face)

#     return face_landmarks, face_rects

# # Function to swap faces for all detected faces in the collage
# def swap_faces(source_image, target_image):
#     source_landmarks, source_rects = get_face_landmarks(source_image)
#     target_landmarks, target_rects = get_face_landmarks(target_image)

#     if source_landmarks is None or len(source_landmarks) == 0:
#         print("No face detected in the source image.")
#         return None
#     if target_landmarks is None or len(target_landmarks) == 0:
#         print("No faces detected in the target image.")
#         return None

#     # Use the first detected face from source as reference
#     source_face = source_image[source_rects[0].top():source_rects[0].bottom(), source_rects[0].left():source_rects[0].right()]
    
#     for rect in target_rects:
#         x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        
#         # Resize source face to match the detected face size
#         resized_source_face = cv2.resize(source_face, (w, h))
        
#         # Replace the detected face in the target image with the resized source face
#         target_image[y:y+h, x:x+w] = resized_source_face

#     return target_image

# # Load input images
# source_image = cv2.imread("source.jpeg")  # Single face image
# target_image = cv2.imread("collage.jpeg")  # Image with multiple faces

# if source_image is None or target_image is None:
#     print("Error loading images.")
# else:
#     result = swap_faces(source_image, target_image)
#     if result is not None:
#         output_path = "face_swapped_collage.jpg"
#         cv2.imwrite(output_path, result)
#         print(f"Face-swapped image saved at: {output_path}")

import cv2
import dlib
import numpy as np
import os

# Verify model files exist
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Required model file not found at {MODEL_PATH}")
    print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

def get_landmarks(image):
    """Improved landmark detection with error handling"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)  # Upsample once
    if not faces:
        return None, None
    try:
        landmarks = predictor(gray, faces[0])
        return np.array([(p.x, p.y) for p in landmarks.parts()], dtype=np.int32)
    except Exception as e:
        print(f"Landmark detection error: {str(e)}")
        return None, None

def get_triangles(image_shape, landmarks):
    """Enhanced triangulation with boundary points"""
    # Add image boundary points
    h, w = image_shape[:2]
    boundary_pts = np.array([[0,0], [w//2,0], [w-1,0],
                           [0,h//2], [w-1,h//2],
                           [0,h-1], [w//2,h-1], [w-1,h-1]])
    all_points = np.vstack([landmarks, boundary_pts])
    
    # Calculate Delaunay triangulation
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    for p in all_points:
        subdiv.insert((float(p[0]), float(p[1])))
    
    triangles = []
    for t in subdiv.getTriangleList():
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        
        # Find indices of original landmarks
        indices = []
        for p in [pt1, pt2, pt3]:
            for i, landmark in enumerate(all_points):
                if abs(landmark[0]-p[0]) < 1 and abs(landmark[1]-p[1]) < 1:
                    indices.append(i)
                    break
        if len(indices) == 3:
            triangles.append(indices)
    
    return [t for t in triangles if max(t) < 68]  # Use only facial points

def warp_face(source, target, src_pts, tgt_pts, triangles):
    """Improved warping with affine transformation"""
    warped = np.zeros_like(target)
    
    for tri in triangles:
        # Get triangle points
        s_tri = src_pts[tri]
        t_tri = tgt_pts[tri]
        
        # Calculate affine transform
        trans = cv2.getAffineTransform(np.float32(s_tri), np.float32(t_tri))
        
        # Warp source triangle
        warped_tri = cv2.warpAffine(source, trans, (target.shape[1], target.shape[0]), None,
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        
        # Create mask
        mask = np.zeros_like(target)
        cv2.fillConvexPoly(mask, np.int32(t_tri), (1, 1, 1))
        
        # Add to final image
        warped = warped * (1 - mask) + warped_tri * mask
    
    return warped

def face_swap(source_path, target_path, output_path):
    """Complete face swap pipeline"""
    # Load images
    source = cv2.imread(source_path)
    target = cv2.imread(target_path)
    if source is None or target is None:
        print("❌ Error loading images")
        return

    # Get landmarks
    src_pts = get_landmarks(source)
    tgt_pts = get_landmarks(target)
    if src_pts is None or tgt_pts is None:
        print("❌ Face detection failed")
        return

    # Get triangulation
    triangles = get_triangles(target.shape, tgt_pts)
    
    # Warp face
    warped = warp_face(source, target, src_pts, tgt_pts, triangles)
    
    # Prepare mask
    mask = np.zeros_like(target)
    cv2.fillConvexPoly(mask, tgt_pts, (255, 255, 255))
    
    # Seamless cloning
    center = np.mean(tgt_pts, axis=0).astype(int)
    result = cv2.seamlessClone(warped, target, mask, tuple(center), cv2.NORMAL_CLONE)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"✅ Successfully saved: {output_path}")

if __name__ == "__main__":
    # Example usage
    face_swap("source.jpeg", "target.jpeg", "output/result.jpg")