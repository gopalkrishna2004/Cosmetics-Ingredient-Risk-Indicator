# Ingredient Inspector: AI-Powered Cosmetics Ingredient Risk Indicator

## Overview
Ingredient Inspector is an AI-driven tool designed to assess the safety of cosmetic products by analyzing their ingredient lists. The application uses AI methods to identify potentially harmful ingredients, highlight them in product labels, and provide detailed side effect information. 

With its intuitive graphical interface, this tool empowers users to make informed decisions about the cosmetic products they use.

## Features
- **Ingredient Detection**: Extracts ingredient lists from product labels using Optical Character Recognition (OCR).
- **Harmful Ingredient Identification**: Compares extracted ingredients against a database of known harmful substances using similarity matching.
- **Side Effect Analysis**: Uses Generative AI (Gemini API) to retrieve concise side effect summaries for each harmful ingredient.
- **Visualization**: Highlights unsafe ingredients directly on the product label image.
- **Image Retrieval**: Fetches representative images of side effects from Google Custom Search API.

## Technologies Used
- **Python**
  - `cv2`: For image processing.
  - `pytesseract`: For OCR to extract text from product labels.
  - `sentence-transformers`: For embedding generation and similarity comparison.
  - `openpyxl`: For loading and managing harmful ingredient data from an Excel file.
  - `google.generativeai`: For generating ingredient side effect summaries.
  - `requests`: For fetching side effect images using Google Custom Search API.
  - `tkinter`: For creating the user-friendly GUI.
  - `PIL`: For image rendering in the GUI.
- **External APIs**
  - Gemini AI API (via `google.generativeai`)
  - Google Custom Search API for image retrieval

## Setup
### Prerequisites
1. Python 3.8 or later.
2. Install the required Python libraries:
    ```bash
    pip install opencv-python pytesseract sentence-transformers openpyxl google-generativeai requests pillow
    ```
3. Install Tesseract OCR:
    - Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).
    - Update the `pytesseract.pytesseract.tesseract_cmd` path in the script to point to the Tesseract executable.
4. Set up API keys:
    - **Gemini API**: Obtain an API key and replace the placeholder in the code.
    - **Google Custom Search API**: Generate an API key and a search engine ID (CX) from [Google Developers Console](https://console.developers.google.com/).

### Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/your-repo/ingredient-inspector.git
    cd Ingredient_lnspector
    ```
2. Place the `harmful_ingredients.xlsx` file in the project directory. This file contains a list of harmful ingredients in column A, starting from row 2.

## Usage
1. Run the application:
    ```bash
    python main.py
    ```
2. Use the GUI to:
    - Upload an image of a product label.
    - View highlighted unsafe ingredients and their side effects.
    - Explore representative images for common side effects.

## Example Workflow
1. **Input**: Upload an image of a cosmetic product label.
2. **Processing**: The application extracts text using OCR, analyzes the ingredient list, and identifies harmful substances.
3. **Output**: Unsafe ingredients are highlighted on the image, and detailed side effect information is displayed with illustrative images.

## Screenshots

### UNSAFE
![INTERFACE](Ingredient_Inspector/Test%20Images/harmful.jpeg)

### SAFE
![INTERFACE](Ingredient_Inspector/Test%20Images/safe.jpeg)

## Acknowledgments
- Tesseract OCR
- Sentence Transformers
- Google Generative AI
- Google Custom Search API
