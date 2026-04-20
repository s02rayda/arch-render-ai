Arch-Render AI
Arch-Render AI is a full-stack architectural visualization tool that transforms sketches into photorealistic interior renders using artificial intelligence. It is designed for architects and designers who require rapid, high-fidelity visualization.
Features
Precision Mode (Canny): Utilizes ControlNet to strictly follow sketch lines, ensuring furniture layout and structural geometry are preserved.
Creative Mode (Text-to-Image): Generates architectural concepts from scratch using natural language prompts.
Memory Optimized: Designed for local execution on consumer GPUs (8GB VRAM) using model offloading techniques.
Modern UI: A responsive interface built with React, Vite, and Tailwind CSS.
Tech Stack
Backend: Python, FastAPI, PyTorch, Diffusers, OpenCV.
Frontend: React, Vite, Tailwind CSS, Axios.
AI Engine: Stable Diffusion 1.5 (Realistic Vision V5.1) + ControlNet (Canny).
Prerequisites
Python 3.10+
Node.js 18+
NVIDIA GPU (Recommended: 8GB+ VRAM)
CUDA Toolkit
Installation
1. Clone the repository
code
Bash
git clone https://github.com/yourusername/arch-render-ai.git
cd arch-render-ai
2. Setup Backend
code
Bash
cd backend
python -m venv venv
# Activate venv (Windows)
venv\Scripts\activate
pip install -r requirements.txt
python main.py
3. Setup Frontend
code
Bash
cd ../frontend
npm install
npm run dev
Technical Implementation
Precision Mode: The backend uses OpenCV Canny Edge Detection to extract structural lines from the input sketch. These lines serve as a conditioning signal for the Stable Diffusion U-Net, ensuring the output adheres to the user's design.
Creative Mode: The backend sets the ControlNet conditioning scale to 0.0, allowing the model to generate images based solely on the provided text prompt.
Memory Management: The application utilizes pipe.enable_model_cpu_offload() to manage VRAM usage, allowing the pipeline to run on hardware with limited memory by offloading inactive components to system RAM.
License
This project is intended for educational and architectural visualization purposes.
