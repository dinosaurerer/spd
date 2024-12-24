import os
import cv2


def create_folders(base_path, class_names):
    """
    Create folders for each class name in the specified base path.
    :param base_path: Path where folders will be created.
    :param class_names: List of class names to create folders for.
    """
    for idx, class_name in enumerate(class_names):
        # Replace invalid characters (e.g., ':') in folder names
        sanitized_folder_name = f"{idx}_{class_name.replace(':', '_')}"
        folder_path = os.path.join(base_path, sanitized_folder_name)
        os.makedirs(folder_path, exist_ok=True)


def capture_images(base_path, class_names):
    """
    Capture images for each class using the webcam.
    :param base_path: Path where images will be stored.
    :param class_names: List of class names corresponding to categories.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    for idx, class_name in enumerate(class_names):
        # Replace invalid characters (e.g., ':') in folder names
        sanitized_folder_name = f"{idx}_{class_name.replace(':', '_')}"
        folder_path = os.path.join(base_path, sanitized_folder_name)
        print(f"Starting collection for category: {class_name}")
        print("Press SPACE to capture an image, 'n' to move to the next category.")

        image_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # Display the frame with instructions
            instruction = f"Capturing: {class_name} - Press SPACE to capture, 'n' for next category."
            frame_copy = frame.copy()
            cv2.putText(frame_copy, instruction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Capture", frame_copy)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # Capture and save the image
                image_path = os.path.join(folder_path, f"{class_name}_{image_count:03d}.jpg")
                cv2.imwrite(image_path, frame)
                image_count += 1
                print(f"Image saved: {image_path}")

            elif key == ord('n'):
                # Move to the next category
                print(f"Finished collecting for category: {class_name}")
                break

        if key == ord('n'):
            continue

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    base_path = os.getcwd()  # Base path for folders
    class_names = [
        "normal",
        "head_tilt_left",
        "head_tilt_right",
        "body_left",
        "body_right",
        "left_support_head",
        "right_support_head",
        "lying_down",
        "head_forward"
    ]

    # Step 1: Create folders
    create_folders(base_path, class_names)

    # Step 2: Capture images
    capture_images(base_path, class_names)

    print("Image collection complete.")
