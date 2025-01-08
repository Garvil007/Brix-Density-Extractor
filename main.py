import os
import cv2
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from pdf2image import convert_from_bytes
import google.generativeai as genai
from PIL import Image
import io

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_KEY"))

def save_intermediate_image(image, step_name, save_folder="intermediate_images"):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    file_path = os.path.join(save_folder, f"{step_name}.jpg")
    cv2.imwrite(file_path, image)
    print(f"Saved intermediate image for {step_name}: {file_path}")

def straighten_and_crop_image(image_data):
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image not found or unable to load.")

    save_intermediate_image(img, "original")

    alpha = 1.2
    beta = 50
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    save_intermediate_image(adjusted_img, "brightness_contrast_adjusted")

    gray = cv2.cvtColor(adjusted_img, cv2.COLOR_BGR2GRAY)
    save_intermediate_image(gray, "grayscale")

    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY_INV, 
        17, 4
    )
    save_intermediate_image(thresh, "adaptive_thresholded")

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image.")

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest_contour = contours[0]

    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped_image = thresh[y:y+h, x:x+w]
    save_intermediate_image(cropped_image, "cropped_contour")

    lines = cv2.HoughLinesP(
        cropped_image, 1, np.pi / 180, threshold=50, 
        minLineLength=50, maxLineGap=10
    )
    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -5 <= angle <= 5:
                detected_lines.append((x1, y1, x2, y2))

    return cropped_image, detected_lines

def cluster_and_draw_middle_line(cropped_image, lines):
    if not lines:
        raise ValueError("No lines detected to cluster.")

    y_coords = []
    for line in lines:
        _, y1, _, y2 = line
        y_coords.append((y1 + y2) // 2)

    y_coords = np.array(y_coords).reshape(-1, 1)

    num_clusters = min(3, len(y_coords))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    kmeans.fit(y_coords)

    image_center_y = cropped_image.shape[0] // 2
    cluster_centers = kmeans.cluster_centers_.flatten()
    middle_cluster_idx = np.argmin(np.abs(cluster_centers - image_center_y))

    middle_cluster_lines = [
        line for i, line in enumerate(lines) if kmeans.labels_[i] == middle_cluster_idx
    ]

    x_coords, y_coords = [], []
    for line in middle_cluster_lines:
        x1, y1, x2, y2 = line
        x_coords.extend([x1, x2])
        y_coords.extend([y1, y2])

    x_coords = np.array(x_coords).reshape(-1, 1)
    y_coords = np.array(y_coords)
    reg = LinearRegression()
    reg.fit(x_coords, y_coords)

    x_start, x_end = 0, cropped_image.shape[1]
    y_start = int(reg.predict([[x_start]])[0])
    y_end = int(reg.predict([[x_end]])[0])

    line_image = cv2.cvtColor(cropped_image, cv2.COLOR_GRAY2BGR)
    cv2.line(line_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
    save_intermediate_image(line_image, "final_horizontal_line")
    return line_image

def upload_to_gemini(file_path, mime_type=None):
    file = genai.upload_file(file_path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file

def create_model():
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
    )
    return model

def analyze_image(file_path, prompt):
    file_uri = [upload_to_gemini(file_path, mime_type="image/jpeg"),]

    model = create_model()

    chat_session = model.start_chat(
        history=[
            {
                "role": "user",
                "parts": [
                    file_uri[0],
                    prompt,
                ],
            }
        ]
    )

    response = chat_session.send_message("Please analyze the image and provide your insights.")
    return response.text

def verify_uploaded_file(file_uri):
    try:
        file_info = genai.get_file(file_uri)
        if file_info:
            print(f"File '{file_info.display_name}' is successfully uploaded.")
            print(f"File URI: {file_info.uri}")
            print(f"MIME Type: {file_info.mime_type}")
        else:
            print("File not found or inaccessible.")
    except Exception as e:
        print(f"Error verifying file: {str(e)}")

if __name__ == "__main__":
    gemini_path = "intermediate_images/final_horizontal_line.jpg"

    image_path = "Urine/Brix_3.5/test_image_16_lens_16.67.jpg"
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    cropped_image, lines = straighten_and_crop_image(image_data)

    if lines:
        processed_image = cluster_and_draw_middle_line(cropped_image, lines)

    analysis_prompt = """
    Analyze the vertical scale in the given image, which ranges from 0 (at the bottom) to 10 (at the top).
    Follow these instructions precisely to determine the exact reading where the horizontal red line intersects the scale:

    1. Identify the whole number immediately below the red line as the reference number.
    2. If the red line passes exactly through the center of a number, that number is the reading.
    3. For cases where the line does not pass exactly through the center:
        - If the red line is slightly above the reference number, add 0.5 to the reference number.
        - If the red line is slightly below the next number (but not centered), add 0.7 to the reference number.
    4. Ensure you observe the numbers on both sides of the scale:
        - Odd numbers are on the left side.
        - Even numbers (including 0) are on the right side.
    5. Double-check the horizontal alignment to ensure the determination is consistent for both the left and right markings.
    """

    try:
        response_text = analyze_image(gemini_path, analysis_prompt)
        print("AI Response:\n", response_text)
    except Exception as e:
        print("An error occurred:", str(e))
