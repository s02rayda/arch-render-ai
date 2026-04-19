Arch-Render AI
Arch-Render AI is a local, full-stack web application designed to assist interior designers and architects in visualizing 3D blockout sketches. By leveraging Stable Diffusion and ControlNet, the application transforms basic 3D geometry into photorealistic architectural renders.
Features
AI-Powered Rendering: Uses Stable Diffusion 1.5 (Realistic Vision V5.1) to generate high-quality architectural visualizations.
Geometric Conditioning: Utilizes ControlNet (Depth) to ensure the generated output respects the spatial layout and furniture placement of the original sketch.
Local Execution: Runs entirely on local hardware, ensuring data privacy and zero cloud-compute costs.
Hardware Optimized: Implements model CPU offloading to run efficiently on consumer-grade GPUs with 8GB VRAM.
Full-Stack Architecture: Features a FastAPI backend for high-performance image processing and a React/Vite frontend for an intuitive user experience.
Tech Stack
Backend: Python, FastAPI, PyTorch, Diffusers, OpenCV.
Frontend: React, Vite, Axios, Tailwind CSS.
AI Models: Realistic Vision V5.1, ControlNet (Depth), MiDaS.
Prerequisites
Python 3.10+
Node.js 18+
NVIDIA GPU with CUDA support (RTX 4060 or equivalent recommended)
CUDA Toolkit 11.8+
Installation
Clone the repository:
code
Bash
git clone https://github.com/s02rayda/arch-render-ai.git
cd arch-render-ai
Setup the Backend:
code
Bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
Setup the Frontend:
code
Bash
cd ../frontend
npm install
npm install axios
Running the Application
Start the Backend:
code
Bash
cd backend
.\venv\Scripts\activate
uvicorn main:app --reload
Start the Frontend:
code
Bash
cd ../frontend
npm run dev
Access the application at the URL provided by the Vite development server (usually http://localhost:5173).
License
This project is open-source and available for educational and professional use.