import { useState } from 'react';
import axios from 'axios';

function App() {
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

  const handleGenerate = async () => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("prompt", styles[style]);

    try {
      const response = await axios.post("http://127.0.0.1:8000/generate", formData, {
        responseType: 'blob' 
      });
      const imageUrl = URL.createObjectURL(new Blob([response.data], { type: 'image/png' }));
      setResult(imageUrl);
    } catch (error) {
      alert("Generation failed!");
    } finally {
      setLoading(false);
    }
  };

  const downloadImage = () => {
    const link = document.createElement('a');
    link.href = result;
    link.download = 'arch-render.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="p-8 bg-slate-950 text-slate-200 min-h-screen font-sans">
      <h1 className="text-4xl font-bold mb-8 text-white">Arch-Render AI</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Controls Panel */}
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 space-y-6">
          <input type="file" onChange={handleFileChange} className="w-full text-sm text-slate-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-600 file:text-white hover:file:bg-indigo-700" />
          
          <div>
            <label className="block mb-2 text-sm font-medium">Select Style</label>
            <select value={style} onChange={(e) => setStyle(e.target.value)} className="w-full p-3 bg-slate-800 rounded-lg border border-slate-700 text-white">
              {Object.keys(styles).map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>

          <button 
            onClick={handleGenerate} 
            disabled={loading || !file}
            className={`w-full py-4 rounded-lg font-bold transition ${loading ? 'bg-slate-700' : 'bg-indigo-600 hover:bg-indigo-500'}`}
          >
            {loading ? "Rendering..." : "Generate Render"}
          </button>
        </div>

        {/* Result Panel */}
        <div className="bg-slate-900 p-6 rounded-xl border border-slate-800 flex flex-col items-center justify-center">
          {loading ? (
            <div className="text-indigo-400 animate-pulse">Processing AI Render...</div>
          ) : result ? (
            <div className="space-y-4 w-full">
              <img src={result} alt="Render" className="rounded-lg shadow-2xl w-full" />
              <button onClick={downloadImage} className="w-full py-2 bg-slate-700 hover:bg-slate-600 rounded-lg">Download Render</button>
            </div>
          ) : (
            <p className="text-slate-500">Upload a sketch to begin</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;