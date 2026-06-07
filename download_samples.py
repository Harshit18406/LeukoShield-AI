import numpy as np
import cv2

print("🔄 Generating synthetic blood cell microscopy test images...")

def create_mock_cell_image(filename, is_malignant=False):
    # Create a purple/pinkish background typical for blood smear stains (Wright-Giemsa stain)
    img = np.ones((400, 400, 3), dtype=np.uint8) * 220
    img[:, :, 0] = 200  # Blue channel bias
    img[:, :, 2] = 230  # Red channel bias
    
    if is_malignant:
        # Malignant cells (Blasts) have huge, irregular nuclei occupying most of the cell
        cv2.circle(img, (200, 200), 110, (180, 140, 150), -1)  # Cell membrane
        cv2.circle(img, (195, 205), 90, (120, 40, 90), -1)    # Massive irregular nucleus
    else:
        # Normal white blood cells have smaller, well-defined/lobed nuclei
        cv2.circle(img, (200, 200), 90, (190, 150, 160), -1)   # Normal cytoplasm
        cv2.circle(img, (180, 200), 35, (100, 30, 80), -1)     # Normal lobe 1
        cv2.circle(img, (220, 210), 30, (100, 30, 80), -1)     # Normal lobe 2

    # Add a few red blood cells in background for realism
    cv2.circle(img, (60, 70), 25, (140, 110, 210), 3)
    cv2.circle(img, (330, 80), 28, (140, 110, 210), 3)
    cv2.circle(img, (80, 320), 26, (140, 110, 210), 3)

    cv2.imwrite(filename, img)
    print(f"✅ Generated and saved: {filename}")

# Generate both types
create_mock_cell_image("malignant_sample.jpg", is_malignant=True)
create_mock_cell_image("normal_sample.jpg", is_malignant=False)

print("\n🎉 Done! Ab tumhare folder me dono images physically create ho chuki hain bina kisi network issue ke. Inhe dashboard par upload karo!")