import cv2
import mediapipe as mp
import numpy as np


class SimpleFaceSwapper:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Face Points
        self.FACE_POINTS = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

    def get_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected")

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        points = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append([x, y])

        return np.array(points, dtype=np.int32)

    def get_face_mask(self, image, landmarks):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        face_points = landmarks[self.FACE_POINTS]
        hull = cv2.convexHull(face_points)
        cv2.fillPoly(mask, [hull], 255)
        return mask

    def get_triangulation(self, landmarks, img_shape):
        h, w = img_shape[:2]

        # Corner punkte für Triangular scheiß
        corners = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
        all_points = np.vstack([landmarks, corners])

        # Trianglar cv2.Subdiv2D (besser laut Chatty)
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)

        for point in all_points:
            subdiv.insert((int(point[0]), int(point[1])))

        triangles = subdiv.getTriangleList()

        # Triangles zu Punkte oder so
        triangle_indices = []
        for triangle in triangles:
            pts = triangle.reshape(3, 2)
            indices = []
            for pt in pts:
                # Suche nach dem nähsten Punkt
                distances = np.sum((all_points - pt) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                indices.append(closest_idx)

            # nur trianlges mit valid indacies
            if len(set(indices)) == 3:  # Sind alle Indices unterschiedlich?
                triangle_indices.append(indices)

        return triangle_indices, all_points

    def warp_triangle(self, src_img, dst_img, src_tri, dst_tri):
        # Bounding rectangles
        r1 = cv2.boundingRect(src_tri)
        r2 = cv2.boundingRect(dst_tri)

        # Was ist valid (ist das P chat?)
        if r1[2] <= 0 or r1[3] <= 0 or r2[2] <= 0 or r2[3] <= 0:
            return

        # OFFSETS
        src_tri_offset = src_tri - [r1[0], r1[1]]
        dst_tri_offset = dst_tri - [r2[0], r2[1]]

        # Face mask oder so kp
        mask = np.zeros((r2[3], r2[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dst_tri_offset), 255)

        # Extract source rectangle kp das hat chatty gemacht
        src_rect = src_img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        if src_rect.size == 0:
            return

        # affine matrix für smoothing oder so glaube ich
        warp_mat = cv2.getAffineTransform(
            src_tri_offset.astype(np.float32),
            dst_tri_offset.astype(np.float32)
        )

        # anwendung affine
        dst_rect = cv2.warpAffine(
            src_rect, warp_mat, (r2[2], r2[3]),
            borderMode=cv2.BORDER_REFLECT_101
        )

        # anwendung blend (blend deez nutzs)
        mask_norm = mask.astype(np.float32) / 255.0
        if len(dst_rect.shape) == 3:
            mask_norm = mask_norm[:, :, np.newaxis]

        dst_region = dst_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

        # Failsafe
        if dst_region.shape == dst_rect.shape:
            blended = dst_rect * mask_norm + dst_region * (1 - mask_norm)
            dst_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = blended.astype(np.uint8)

    def color_match(self, source_img, target_img, source_landmarks, target_landmarks):
        # Get facial (heh) masks
        source_mask = self.get_face_mask(source_img, source_landmarks)
        target_mask = self.get_face_mask(target_img, target_landmarks)

        # Convert to LAB color space weil obama schwarz ist
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Color chaneel matching wie auf tinder
        for i in range(3):
            source_pixels = source_lab[:, :, i][source_mask > 0]
            target_pixels = target_lab[:, :, i][target_mask > 0]

            if len(source_pixels) > 0 and len(target_pixels) > 0:
                source_mean = np.mean(source_pixels)
                source_std = np.std(source_pixels)
                target_mean = np.mean(target_pixels)
                target_std = np.std(target_pixels)

                if source_std > 0:
                    # Apply color correction (BLACKED)
                    corrected_pixels = ((source_pixels - source_mean) *
                                        (target_std / source_std) + target_mean)
                    source_lab[:, :, i][source_mask > 0] = corrected_pixels

        # bitte zurück bitti zu BRG
        corrected = cv2.cvtColor(np.clip(source_lab, 0, 255).astype(np.uint8),
                                 cv2.COLOR_LAB2BGR)
        return corrected

    def create_blend_mask(self, image, landmarks):
        # Create base mask
        mask = self.get_face_mask(image, landmarks)

        # gaus? nicht der nuttensohn (apply blurring)
        mask_blurred = cv2.GaussianBlur(mask, (21, 21), 0)

        # normalisieren
        return mask_blurred.astype(np.float32) / 255.0

    def swap_faces(self, source_img, target_img):
        # resizing (bitte mach mein penis größer)
        target_h, target_w = target_img.shape[:2]
        source_resized = cv2.resize(source_img, (target_w, target_h),
                                    interpolation=cv2.INTER_LANCZOS4)

        # facial (heh) landmarks
        source_landmarks = self.get_landmarks(source_resized)
        target_landmarks = self.get_landmarks(target_img)

        # Tinder color matching
        source_corrected = self.color_match(source_resized, target_img,
                                            source_landmarks, target_landmarks)

        # dreicke
        triangles, all_points = self.get_triangulation(target_landmarks, target_img.shape)

        # output definen und so
        output = target_img.copy().astype(np.float32)

        """
        Also den Teil hier hat KI gemacht kein plan was hier abgeht irgendwas mit warping und so. Aber so smart bin ich
        nicht kein bock das zu checken
        """

        for triangle_indices in triangles:
            try:
                # Ensure we have valid triangle with source landmarks
                if (len(triangle_indices) == 3 and
                        all(idx < len(source_landmarks) for idx in triangle_indices)):
                    src_tri = source_landmarks[triangle_indices]
                    dst_tri = all_points[triangle_indices]

                    self.warp_triangle(
                        source_corrected.astype(np.float32),
                        output,
                        src_tri.astype(np.float32),
                        dst_tri.astype(np.float32)
                    )
            except (IndexError, ValueError):
                continue

        # Smoothie machen und so mit Mask
        blend_mask = self.create_blend_mask(target_img, target_landmarks)
        mask_3d = blend_mask[:, :, np.newaxis]

        # Finaler mixer (wixxer)
        result = (output * mask_3d +
                  target_img.astype(np.float32) * (1 - mask_3d))

        return np.clip(result, 0, 255).astype(np.uint8)