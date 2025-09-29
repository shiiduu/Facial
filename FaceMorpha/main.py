import cv2
import os
from simple_face_swapper import SimpleFaceSwapper

"""""
Ich bin ganz ehrlich digga kein plan wie CV funktioniert hab das alles von github geklaut und bisschen angepasst
Also wenn du plan hast und das verbessern willst nur zu. Ich habs nur so hingekackt dass es läuft
Achso und komplett main.py ki generiert frag mich hier zu gar nix. 
"""""


def main():
    swapper = SimpleFaceSwapper()

    source_path = "images/primary.jpg"
    target_path = "images/secondary.jpg"

    if not os.path.exists(source_path) or not os.path.exists(target_path):
        print("Add primary.jpg and secondary.jpg to images/ folder")
        return

    source_img = cv2.imread(source_path)
    target_img = cv2.imread(target_path)

    if source_img is None or target_img is None:
        print("Could not load images")
        return

    try:
        print("Processing face swap...")
        result = swapper.swap_faces(source_img, target_img)

        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/result.jpg", result)
        print("Saved: output/result.jpg")

        # Display result
        cv2.imshow("Face Swap Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()