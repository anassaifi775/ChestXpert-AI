import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, render_template, flash, redirect, url_for, jsonify
from dotenv import load_dotenv # Import dotenv
import traceback

# --- Configuration ---
# Set the Hugging Face cache directory to the external drive BEFORE importing transformers
os.environ['HF_HOME'] = '/Volumes/Backup/huggingface_cache'
# Set TMPDIR to external drive to avoid "No space left on device" errors on main drive
os.environ['TMPDIR'] = '/Volumes/Backup/tmp'

# Import necessary classes from transformers
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
)

load_dotenv() # Load environment variables from .env file

BLIP_MODEL_ID = "anassaifi8912/chestxray-blip-report-generator"
QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN") # Get token from env

if not HF_TOKEN:
    print("Warning: HUGGING_FACE_HUB_TOKEN environment variable not set. Model downloads might fail if they are gated.")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Fallback to cuda if mps is not there but cuda is (unlikely on Mac M1 but good practice)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

print(f"Using device: {DEVICE}")

# Force BLIP to CPU to avoid "MPS does not support cumsum op with int64 input" error
BLIP_DEVICE = torch.device("cpu") 
# Force Qwen to CPU for the same reason
QWEN_DEVICE = torch.device("cpu") 

UPLOAD_FOLDER = 'uploads' # Optional: If you want to save uploads temporarily
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- Global Variables for Model Components ---
blip_model = None
blip_processor = None
qwen_model = None
qwen_tokenizer = None

def load_blip_model_components():
    """Loads the BLIP model and processor."""
    global blip_model, blip_processor
    try:
        print(f"Loading BLIP model ({BLIP_MODEL_ID}) components on device: {BLIP_DEVICE}")
        
        blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
        blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL_ID)
        
        blip_model.to(BLIP_DEVICE)
        blip_model.eval()
        print("BLIP Model loaded successfully.")

    except Exception as e:
        print(f"Error loading BLIP model components: {e}")
        raise

def load_qwen_model_components():
    """Loads the Qwen model and tokenizer."""
    global qwen_model, qwen_tokenizer
    try:
        print(f"Loading Qwen model ({QWEN_MODEL_ID}) components on device: {QWEN_DEVICE}...")
        
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID, token=HF_TOKEN)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.float32, # CPU doesn't support float16 well
            device_map=None, # Manually handling device
            token=HF_TOKEN
        )
        qwen_model.to(QWEN_DEVICE)
            
        qwen_model.eval()
        print("Qwen Model and Tokenizer loaded successfully.")

    except Exception as e:
        print(f"Error loading Qwen model components: {e}")
        # Decide if the app should run without the chat feature or crash
        qwen_model = None
        qwen_tokenizer = None
        print("WARNING: Chatbot functionality will be disabled due to loading error.")
        # raise # Uncomment this if the chat feature is critical

# --- Inference Function (BLIP) ---
def generate_report(image_bytes, max_length=100):
    """Generates a report/caption for the given image bytes using BLIP."""
    global blip_model, blip_processor
    if not all([blip_model, blip_processor]):
         load_blip_model_components() # Attempt to load again if missing
         if not all([blip_model, blip_processor]):
             raise RuntimeError("BLIP model components failed to load.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Ensure image is properly sized (BLIP expects certain dimensions)
        # Resize to 384x384 as used in training
        image = image.resize((384, 384), Image.Resampling.LANCZOS)
        
        # Prepare inputs
        inputs = blip_processor(images=image, return_tensors="pt").to(BLIP_DEVICE)

        # Generate with better parameters for medical reports
        with torch.no_grad():
            generated_ids = blip_model.generate(
                **inputs,
                max_length=max_length,
                min_length=20,  # Ensure some minimum length
                num_beams=6,  # Increased for better quality
                repetition_penalty=1.2,  # Reduced to allow some repetition
                length_penalty=1.0,  # Neutral length penalty
                early_stopping=True,
                do_sample=False,  # Deterministic generation
                temperature=1.0
            )
            report = blip_processor.decode(generated_ids[0], skip_special_tokens=True)
            
            # Clean up the report
            report = clean_medical_report(report)
            return report

    except Exception as e:
        print(f"Error during BLIP report generation: {e}")
        return f"Error generating report: {e}"

def clean_medical_report(report):
    """Clean and improve the generated medical report."""
    # Remove excessive repetition
    sentences = report.split('. ')
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and sentence.lower() not in seen:
            unique_sentences.append(sentence)
            seen.add(sentence.lower())
    
    cleaned = '. '.join(unique_sentences)
    
    # Ensure it ends with a period
    if cleaned and not cleaned.endswith('.'):
        cleaned += '.'
    
    # If too short, add a disclaimer
    if len(cleaned.split()) < 10:
        cleaned += " Please consult a radiologist for proper interpretation."
    
    return cleaned

# --- Chat Function (Qwen) ---
def generate_chat_response(question, report_context, max_new_tokens=250):
    """Generates a chat response using Qwen based on the report context."""
    global qwen_model, qwen_tokenizer
    if not qwen_model or not qwen_tokenizer:
        return "Chatbot is currently unavailable."

    # System prompt to guide the LLM
    system_prompt = "You are a helpful medical assistant. I'm a medical student, your task is to help me understand the following chest X-ray report."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Here is the report:\n{report_context}\n\nMy Question: {question}"}
    ]

    try:
        text = qwen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)

        with torch.no_grad():
            generated_ids = qwen_model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    except Exception as e:
        print(f"Error during Qwen chat generation: {e}")
        return f"Error text generation: {e}"


# --- Flask Application Setup ---
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load models when the application starts
print("Loading models on application startup...")
try:
    load_blip_model_components()
    load_qwen_model_components()
    print("Model loading complete.")
except Exception as e:
    print(f"FATAL ERROR during model loading: {e}")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---- Function to Parse Filename ----
def parse_patient_info(filename):
    """
    Parses a filename like '00069-34-Frontal-AP-63.0-Male-White.png'
    Returns a dictionary with 'view', 'age', 'gender', 'ethnicity'.
    Returns None if parsing fails.
    """
    try:
        base_name = os.path.splitext(filename)[0]
        parts = base_name.split('-')
        # Expected structure based on example: ... - ViewPart1 - ViewPartN - Age - Gender - Ethnicity
        if len(parts) < 5: 
            # print(f"Warning: Filename '{filename}' has fewer parts than expected.")
            return None

        ethnicity = parts[-1]
        gender = parts[-2]
        age_str = parts[-3]
        try:
            age = int(float(age_str))
        except ValueError:
            return None 

        view_parts = parts[2:-3]
        view = '-'.join(view_parts) if view_parts else "Unknown" 

        return {
            'view': view,
            'age': age,
            'gender': gender.capitalize(), 
            'ethnicity': ethnicity.capitalize() 
        }
    except Exception as e:
        print(f"Error parsing filename '{filename}': {e}")
        return None

# --- Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    chatbot_available = bool(qwen_model and qwen_tokenizer)
    return render_template('index.html', chatbot_available=chatbot_available)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and prediction."""
    chatbot_available = bool(qwen_model and qwen_tokenizer)
    patient_info = None

    if 'image' not in request.files:
        flash('No image file part in the request.', 'danger')
        return redirect(url_for('index'))

    file = request.files['image']
    
    try:
        max_length = int(request.form.get('max_length', 100))
        if not (10 <= max_length <= 512):
            raise ValueError("Max length must be between 10 and 512.")
    except ValueError as e:
         flash(f'Invalid Max Length value: {e}', 'danger')
         return redirect(url_for('index'))

    if file.filename == '':
        flash('No image selected for uploading.', 'warning')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        try:
            image_bytes = file.read()

            original_filename = file.filename
            patient_info = parse_patient_info(original_filename)

            # Generate report using BLIP
            report = generate_report(image_bytes, max_length)

            if report.startswith("Error"):
                 flash(f'Report Generation Failed: {report}', 'danger')
                 image_data = base64.b64encode(image_bytes).decode('utf-8')
                 return render_template('index.html',
                                        report=None,
                                        image_data=image_data,
                                        patient_info=patient_info,
                                        chatbot_available=chatbot_available)

            image_data = base64.b64encode(image_bytes).decode('utf-8')

            return render_template('index.html',
                                   report=report,
                                   image_data=image_data,
                                   patient_info=patient_info,
                                   chatbot_available=chatbot_available)

        except RuntimeError as rt_error:
            flash(f'Model loading error: {rt_error}. Please check server logs.', 'danger')
            print(f"Runtime error during prediction: {rt_error}\n{traceback.format_exc()}")
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'An unexpected error occurred during prediction: {e}', 'danger')
            print(f"Error during prediction: {e}\n{traceback.format_exc()}")
            return redirect(url_for('index'))
    else:
        flash('Invalid image file type. Allowed types: png, jpg, jpeg.', 'danger')
        return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests based on the generated report."""
    if not qwen_model or not qwen_tokenizer:
        return jsonify({"answer": "Chatbot is not available."}), 503 

    data = request.get_json()
    if not data or 'question' not in data or 'report_context' not in data:
        return jsonify({"error": "Missing question or report context"}), 400

    question = data['question']
    report_context = data['report_context']

    try:
        answer = generate_chat_response(question, report_context)
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": "Failed to generate chat response"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)