import cv2
import openpyxl
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
import pytesseract
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import requests
import io

genai.configure(api_key='gemini api')

# Initialize the generative model
model = genai.GenerativeModel('gemini-1.5-flash')

# Path to tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load harmful ingredients from the Excel file
wb = openpyxl.load_workbook('harmful_ingredients.xlsx')
sheet = wb.active

# Load pre-trained sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings for harmful ingredients
harmful_ingredients_list = [row[0].lower() for row in sheet.iter_rows(min_row=2, values_only=True)]
harmful_embeddings = embedding_model.encode(harmful_ingredients_list, convert_to_tensor=True)

def get_ingredient_side_effects(ingredient):
    """Retrieve side effects for a given ingredient using Gemini API."""
    try:
        prompt = f"Provide a concise summary of potential side effects for the cosmetic ingredient {ingredient} in 2-3 sentences. Focus medical information, give negative information, give side effects"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Unable to retrieve information for {ingredient}. Error: {str(e)}"

def state_space_search(product_ingredients, threshold=0.8):
    """Explore the state space to find matches for harmful ingredients."""
    unsafe_ingredients = []

    # Generate embeddings for product ingredients
    product_embeddings = embedding_model.encode(product_ingredients, convert_to_tensor=True)

    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(product_embeddings, harmful_embeddings)

    # Traverse the state space
    for i, product_ingredient in enumerate(product_ingredients):
        for j, harmful_ingredient in enumerate(harmful_ingredients_list):
            similarity = similarity_matrix[i][j]
            if similarity >= threshold:
                # Match found
                unsafe_ingredients.append(harmful_ingredient)
                break  # Stop further exploration for this ingredient

    return list(set(unsafe_ingredients))  # Return unique matches

def check_safety(product_text):
    """Check for unsafe ingredients in the OCR-extracted product text."""
    product_ingredients = [ingredient.strip().lower() for ingredient in product_text.split(',')]
    unsafe_ingredients = state_space_search(product_ingredients)

    # Filter only those unsafe ingredients that are present in the OCR output
    confirmed_unsafe = [
        ingredient for ingredient in unsafe_ingredients
        if any(ingredient in product for product in product_ingredients)
    ]

    return confirmed_unsafe

def highlight_ingredients(image, unsafe_ingredients):
    """Highlight harmful ingredients as a single block in the image."""
    highlighted_image = image.copy()

    # Get OCR data with bounding boxes
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Reconstruct text line by line
    lines = {}
    for i, word in enumerate(data['text']):
        if word.strip():  # Ignore empty words
            line_num = data['line_num'][i]
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(i)  # Store word index for this line

    # Loop through each ingredient
    for ingredient in unsafe_ingredients:
        ingredient_lower = ingredient.lower()
        for line_indices in lines.values():
            # Reconstruct the line text and indices
            line_text = " ".join(data['text'][i] for i in line_indices).lower()

            # Check if the ingredient exists in the line
            if ingredient_lower in line_text:
                start_idx = line_text.find(ingredient_lower)  # Start position
                end_idx = start_idx + len(ingredient_lower)  # End position

                # Highlight all bounding boxes corresponding to the ingredient
                word_positions = []
                current_position = 0
                for i in line_indices:
                    word = data['text'][i]
                    word_length = len(word)
                    # Check if the word is part of the ingredient match
                    if current_position <= start_idx < current_position + word_length or \
                            current_position <= end_idx <= current_position + word_length:
                        word_positions.append(i)
                    current_position += word_length + 1  # +1 for spaces

                # Combine bounding boxes for the matched words
                if word_positions:
                    x_min = min(data['left'][i] for i in word_positions)
                    y_min = min(data['top'][i] for i in word_positions)
                    x_max = max(data['left'][i] + data['width'][i] for i in word_positions)
                    y_max = max(data['top'][i] + data['height'][i] for i in word_positions)

                    # Draw a single rectangle for the combined bounding box
                    overlay = highlighted_image.copy()
                    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, highlighted_image, 0.7, 0, highlighted_image)
                    cv2.rectangle(highlighted_image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

                    # Add ingredient label
                    label_text = ingredient.upper()
                    (text_width, text_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(
                        highlighted_image,
                        (x_min, y_min - text_height - 10),
                        (x_min + text_width, y_min - 2),
                        (255, 255, 255),
                        -1,
                    )
                    cv2.putText(
                        highlighted_image,
                        label_text,
                        (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )

    return highlighted_image

def resize_image(image, max_size):
    """Resize image while maintaining aspect ratio."""
    height, width = image.shape[:2]
    ratio = 1
    dimensions = (width, height)

    if height > width:
        if height > max_size:
            ratio = max_size / height
            dimensions = (int(width * ratio), max_size)
    else:
        if width > max_size:
            ratio = max_size / width
            dimensions = (max_size, int(height * ratio))
    
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

def cv2_to_tk(cv2_image):
    """Convert OpenCV image to tkinter PhotoImage."""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return ImageTk.PhotoImage(pil_image)

def get_ingredient_side_effects_list(ingredient):
    """Retrieve side effects for a given ingredient using Gemini API."""
    try:
        prompt = (
            f"List the potential side effects of the cosmetic ingredient '{ingredient}' in a single-word or very short format, "
            f"don't include allergy, separated by commas. Only include negative effects. Example format: rashes, irritation, redness."
        )
        response = model.generate_content(prompt)
        # Parse response into a list of side effects
        effects = [effect.strip() for effect in response.text.strip().split(',')]
        return effects
    except Exception as e:
        return f"Unable to retrieve information for {ingredient}. Error: {str(e)}"

def get_google_images(query, api_key, cx):
    """Retrieve image URLs from Google Custom Search API for a given query."""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={cx}&searchType=image&key={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        results = response.json().get('items', [])
        images = [item['link'] for item in results]
        return images
    else:
        print(f"Error: {response.status_code}")
        return []

google_api_key = "google search api"
google_cx = "04d022967f9404c32"

def fetch_images_for_side_effects(side_effects_list, api_key, cx):
    """Fetch one image for each side effect."""
    images_dict = {}
    for effect in side_effects_list:
        query = f"skin {effect}"  # Prepend 'skin' to each side effect
        images = get_google_images(query, api_key, cx)
        if images:
            images_dict[effect] = images[0]  # Store only the first image URL
    return images_dict


def process_image():
    """Process the image, extract text, and identify unsafe ingredients."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if not file_path:
        result_label.config(text="No image selected", fg="#FF5733")
        return

    # Read and process the image
    image = cv2.imread(file_path)
    display_image = resize_image(image.copy(), 600)
    
    # Extract text and check for unsafe ingredients
    product_text = pytesseract.image_to_string(display_image)
    unsafe_ingredients = check_safety(product_text)

    # Create highlighted version
    highlighted_image = highlight_ingredients(display_image, unsafe_ingredients)
    tk_highlighted = cv2_to_tk(highlighted_image)
    
    # Update image label
    image_label.config(image=tk_highlighted)
    image_label.image = tk_highlighted  # Keep a reference

    # Clear any previous side effect images
    for widget in side_effects_frame.winfo_children():
        widget.destroy()

    if unsafe_ingredients:
        result_label.config(
            text="⚠️ WARNING: The Product Contains UNSAFE Ingredients!",
            fg="#FF5733",
            font=("Arial", 14, "bold")
        )
        
        # Prepare detailed ingredient information
        ingredients_details = []
        all_side_effects_list = []
        for ingredient in unsafe_ingredients:
            side_effects = get_ingredient_side_effects(ingredient)
            side_effects_list = get_ingredient_side_effects_list(ingredient)
            ingredients_details.append(f"• {ingredient.upper()}\n   Side Effects: {side_effects}")
            all_side_effects_list.extend(side_effects_list)
        
        ingredients_text = "\n\n".join(ingredients_details)
        
        definitions_label.config(
            text=f"Harmful Ingredients and Their Potential Side Effects:\n\n{ingredients_text}",
            fg="black",
            font=("Arial", 10),
            wraplength=300,  # Add word wrap
            justify="left"
        )
        
        # Fetch side effect images
        side_effect_images = fetch_images_for_side_effects(list(set(all_side_effects_list)), google_api_key, google_cx)

        # Display each side effect with its corresponding image
        for effect, image_url in side_effect_images.items():
            try:
                # Download the image
                response = requests.get(image_url)
                img_data = response.content
                img = Image.open(io.BytesIO(img_data))
                
                # Resize image to fit the frame
                img.thumbnail((200, 200))  # Resize while maintaining aspect ratio
                photo = ImageTk.PhotoImage(img)
                
                # Create a frame for each side effect
                effect_frame = tk.Frame(side_effects_frame, bg="#ECF0F1")
                effect_frame.pack(pady=5)
                
                # Side effect label
                effect_label = tk.Label(effect_frame, text=effect.capitalize(), bg="#ECF0F1", font=("Arial", 10))
                effect_label.pack()
                
                # Image label
                img_label = tk.Label(effect_frame, image=photo, bg="#ECF0F1")
                img_label.image = photo  # Keep a reference
                img_label.pack()
                
            except Exception as e:
                print(f"Error loading image for {effect}: {e}")
        
    else:
        result_label.config(
            text="✅ This Product is SAFE to Use",
            fg="#2ECC71",
            font=("Arial", 14, "bold")
        )
        definitions_label.config(text="No harmful ingredients detected.", fg="black")

        # Ensure side effects frame is cleared
        side_effects_title.config(text="")

# GUI Setup
window = tk.Tk()
window.title("Ingredient Inspector")
window.geometry("1200x900")  # Increased window size
window.configure(bg="#ECF0F1")

# Header 
header_frame = tk.Frame(window, bg="#3498DB")
header_frame.pack(fill="x")

title_label = tk.Label(
    header_frame, 
    text="Ingredient Inspector", 
    font=("Helvetica", 24, "bold"), 
    bg="#3498DB", 
    fg="white"
)
title_label.pack(pady=10)

# Main content area
content_frame = tk.Frame(window, bg="#ECF0F1")
content_frame.pack(fill="both", expand=True, padx=20, pady=10)

# Left side - Image
image_frame = tk.Frame(content_frame, bg="#ECF0F1")
image_frame.pack(side="left", fill="both", expand=True, padx=10)

# Highlighted image
image_label = tk.Label(image_frame, bg="#ECF0F1")
image_label.pack(pady=5)

# Middle - Analysis
analysis_frame = tk.Frame(content_frame, bg="#ECF0F1", width=300)
analysis_frame.pack(side="left", fill="y", padx=10)

result_label = tk.Label(analysis_frame, font=("Arial", 16), bg="#ECF0F1")
result_label.pack(pady=10)

definitions_label = tk.Label(analysis_frame, font=("Arial", 12), bg="#ECF0F1", justify="left")
definitions_label.pack(pady=10)

# Right side - Side Effects Images
side_effects_container = tk.Frame(content_frame, bg="#ECF0F1", width=300)
side_effects_container.pack(side="right", fill="y", padx=10)

side_effects_title = tk.Label(
    side_effects_container, 
    text="Side Effects Visualization", 
    font=("Arial", 14, "bold"), 
    bg="#ECF0F1"
)
side_effects_title.pack(pady=10)


# Scrollable frame for side effects images
side_effects_canvas = tk.Canvas(side_effects_container, bg="#ECF0F1")
side_effects_scrollbar = ttk.Scrollbar(side_effects_container, orient="vertical", command=side_effects_canvas.yview)
side_effects_frame = tk.Frame(side_effects_canvas, bg="#ECF0F1")

side_effects_frame.bind(
    "<Configure>",
    lambda e: side_effects_canvas.configure(
        scrollregion=side_effects_canvas.bbox("all")
    )
)

side_effects_canvas.create_window((0, 0), window=side_effects_frame, anchor="nw")
side_effects_canvas.configure(yscrollcommand=side_effects_scrollbar.set)

side_effects_canvas.pack(side="left", fill="both", expand=True)
side_effects_scrollbar.pack(side="right", fill="y")

# Footer
footer_frame = tk.Frame(window, bg="#34495E")
footer_frame.pack(fill="x", pady=5)

upload_button = ttk.Button(
    footer_frame, 
    text="Upload Image", 
    command=process_image
)
upload_button.pack(pady=10)

window.mainloop()
