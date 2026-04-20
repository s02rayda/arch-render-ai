import { useState } from 'react';
import axios from 'axios';

function App() {
  const [customPrompt, setCustomPrompt] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [style, setStyle] = useState("Modern");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const styles = {
    Modern: "photorealistic modern living room, minimalist, natural light, 8k, highly detailed",
    Industrial: "industrial loft interior, exposed brick, dark metal furniture, warm lighting, 8k, highly detailed",
    Scandinavian: "scandinavian interior design, light wood, white walls, cozy textiles, bright natural light, 8k, highly detailed"
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

// Add this state
const [mode, setMode] = useState('precision'); // 'precision' or 'creative'

// Update handleGenerate
const handleGenerate = async () => {
  setLoading(true);
  const formData = new FormData();
  
  // Use customPrompt if Creative, otherwise use the style preset
  const finalPrompt = mode === 'creative' ? customPrompt : styles[style];
  formData.append("prompt", finalPrompt);

  let endpoint = "http://127.0.0.1:8000/generate-text-to-image";
  
  if (mode === 'precision') {
    if (!file) { alert("Please upload a file"); setLoading(false); return; }
    formData.append("file", file);
    endpoint = "http://127.0.0.1:8000/generate";
  }

  try {
    const response = await axios.post(endpoint, formData);
    const { job_id } = response.data;
    const imageResponse = await axios.get(`http://127.0.0.1:8000/outputs/${job_id}/render`, { responseType: 'blob' });
    setResult(URL.createObjectURL(imageResponse.data));
  } catch (error) {
    console.error(error);
    alert("Generation failed!");
  } finally {
    setLoading(false);
  }
};

// In your JSX, wrap the file input:
{mode === 'precision' && (
  <input type="file" onChange={handleFileChange} className="..." />
)}

  const downloadImage = () => {
    const link = document.createElement('a');
    link.href = result;
    link.download = 'arch-render.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  return (
    <div className="p-8 bg-slate-50 text-slate-800 min-h-screen font-sans">
      <h1 className="text-4xl font-bold mb-8 text-slate-900">Arch-Render AI</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Controls Panel */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-6">
          
          {/* Toggle */}
          <div className="flex bg-slate-100 p-1 rounded-lg">
            <button 
              onClick={() => setMode('precision')}
              className={`flex-1 py-2 rounded-md transition ${mode === 'precision' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-600'}`}
            >Precision</button>
            <button 
              onClick={() => setMode('creative')}
              className={`flex-1 py-2 rounded-md transition ${mode === 'creative' ? 'bg-indigo-600 text-white shadow-md' : 'text-slate-600'}`}
            >Creative</button>
          </div>

          {mode === 'precision' && (
            <input 
              type="file" 
              onChange={handleFileChange} 
              className="w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-100 file:text-indigo-700 hover:file:bg-indigo-200" 
            />
          )}
          
          {mode === 'precision' ? (
            <div>
              <label className="block mb-2 text-sm font-semibold text-slate-700">Select Style</label>
              <select value={style} onChange={(e) => setStyle(e.target.value)} className="w-full p-3 bg-slate-50 rounded-lg border border-slate-200 text-slate-800">
                {Object.keys(styles).map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          ) : (
            <div>
              <label className="block mb-2 text-sm font-semibold text-slate-700">Describe your vision</label>
              <textarea 
                value={customPrompt}
                onChange={(e) => setCustomPrompt(e.target.value)}
                placeholder="e.g. A bright, airy living room with floor-to-ceiling windows..."
                className="w-full p-3 bg-slate-50 rounded-lg border border-slate-200 text-slate-800 h-32"
              />
            </div>
          )}

          <button 
            onClick={handleGenerate} 
            disabled={loading || (mode === 'precision' && !file)}
            className={`w-full py-4 rounded-xl font-bold transition shadow-lg ${loading ? 'bg-slate-300' : 'bg-indigo-600 hover:bg-indigo-700 text-white'}`}
          >
            {loading ? "Rendering..." : "Generate Render"}
          </button>
        </div>

        {/* Result Panel */}
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col items-center justify-center">
          {loading ? (
            <div className="text-indigo-600 font-medium animate-pulse">Creating your masterpiece...</div>
          ) : result ? (
            <div className="space-y-4 w-full">
              <img src={result} alt="Render" className="rounded-xl shadow-md w-full" />
              <button onClick={downloadImage} className="w-full py-3 bg-slate-100 hover:bg-slate-200 text-slate-800 font-semibold rounded-xl transition">Download Render</button>
            </div>
          ) : (
            <p className="text-slate-400">Upload a sketch or type a prompt to begin</p>
          )}
        </div>
      </div>
    </div>
  );    
}
export default App;