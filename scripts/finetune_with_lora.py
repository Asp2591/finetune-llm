

# Step 2: Import the required libraries
import os
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import load_dotenv

# Step 3: Load environment variables from the .env file
load_dotenv()  # Load environment variables from the .env file

# Retrieve Kaggle credentials from the environment variables
kaggle_username = os.getenv('KAGGLE_USERNAME')
kaggle_key = os.getenv('KAGGLE_KEY')

# Ensure that the Kaggle credentials are available
if not kaggle_username or not kaggle_key:
    raise ValueError("KAGGLE_USERNAME or KAGGLE_KEY environment variables are not set. Please set them in the .env file.")

# Set up Kaggle API authentication using the loaded credentials
os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_key

# Step 4: Authenticate with Kaggle API
api = KaggleApi()
api.authenticate()

# Define the dataset name and path to save it
dataset_name = "varshitanalluri/crop-recommendation-dataset"
dataset_path = "crop-recommendation-dataset"

# Download and unzip the dataset
api.dataset_download_files(dataset_name, path=dataset_path, unzip=True)
print("Path to dataset files:", os.path.abspath(dataset_path))

# Step 5: Load and Prepare Your Data
from datasets import load_dataset

# Load your custom data into a Hugging Face dataset format
dataset = load_dataset('csv', data_files={'train': os.path.join(dataset_path, 'crop_recommendation.csv')})

# Step 6: Load the Pretrained Model
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"  # Choose a base model, e.g., GPT-2 for text generation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Step 7: Apply LoRA Using the `peft` Library
from peft import get_peft_model, LoraConfig

# Configure LoRA for the model
lora_config = LoraConfig(
    r=4,  # LoRA rank
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.05,  # Dropout rate
    target_modules=["c_attn"],  # Targeted layers for LoRA
)

# Wrap the model with LoRA
model = get_peft_model(model, lora_config)

# Step 8: Preprocess and Tokenize the Dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Step 9: Set Up the Training Loop
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora-finetuned-model",
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Set up the Trainer with the tokenized dataset and model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# Step 10: Fine-Tune the Model
trainer.train()

# Step 11: Save the Fine-Tuned Model
model.save_pretrained("./models/finetuned-model")
tokenizer.save_pretrained("./models//finetuned-model")
