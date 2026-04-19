import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx' // This matches the export default App
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)