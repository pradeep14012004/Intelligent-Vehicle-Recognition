import { useState } from 'react'
import {
  CarFront, Upload, ScanLine, Cpu, Image as ImageIcon,
  Activity, CheckCircle2, AlertTriangle, Loader2,
  Gauge, Layers3, Sparkles, RotateCcw
} from 'lucide-react'

const API = 'http://127.0.0.1:8000'

function imageUrl(base64) {
  return base64 ? `data:image/jpeg;base64,${base64}` : null
}

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [preprocessing, setPreprocessing] = useState('none')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [results, setResults] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [dragging, setDragging] = useState(false)

  const selectFile = (selected) => {
    if (!selected) return
    if (!['image/jpeg', 'image/png'].includes(selected.type)) {
      setError('Please select a JPG or PNG image.')
      return
    }
    setFile(selected)
    setPreview(URL.createObjectURL(selected))
    setResults(null)
    setComparison(null)
    setError('')
  }

  const recognize = async () => {
    if (!file) {
      setError('Upload a vehicle image first.')
      return
    }

    setLoading(true)
    setError('')
    setResults(null)

    try {
      const form = new FormData()
      form.append('file', file)
      form.append('preprocessing', preprocessing)

      const response = await fetch(`${API}/api/recognize`, {
        method: 'POST',
        body: form
      })

      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Recognition failed.')
      setResults(data)
    } catch (err) {
      setError(
        err.message.includes('fetch')
          ? 'Cannot connect to FastAPI. Make sure the backend is running on port 8000.'
          : err.message
      )
    } finally {
      setLoading(false)
    }
  }

  const compareMethods = async () => {
    if (!file) return

    setCompareLoading(true)
    setError('')
    setComparison(null)

    try {
      const form = new FormData()
      form.append('file', file)

      const response = await fetch(`${API}/api/compare`, {
        method: 'POST',
        body: form
      })

      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Comparison failed.')

      setComparison(data.comparisons)
    } catch (err) {
      setError(err.message)
    } finally {
      setCompareLoading(false)
    }
  }

  const reset = () => {
    setFile(null)
    setPreview(null)
    setResults(null)
    setComparison(null)
    setError('')
  }

  const detections = results?.detections || []

  return (
    <div className="min-h-screen bg-[#080d18] text-slate-100">
      <header className="border-b border-white/10 bg-[#0b1220]/90">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-cyan-400/10 p-3 text-cyan-300">
              <CarFront size={27} />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-wide">AUTOVISION AI</h1>
              <p className="text-xs text-slate-500">Intelligent Fine-Grained Vehicle Recognition</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-2 text-xs text-emerald-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Recognition platform
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-9">
        <section className="mb-9">
          <p className="mb-3 text-xs font-semibold uppercase tracking-[0.25em] text-cyan-400">
            Computer Vision · Deep Learning · Hybrid AI
          </p>
          <h2 className="text-3xl font-bold tracking-tight md:text-4xl">
            Vehicle Recognition <span className="text-cyan-400">Studio</span>
          </h2>
          <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-400">
            Detect vehicles, identify make and model, and inspect model confidence
            using an integrated YOLO and hybrid feature-fusion pipeline.
          </p>
        </section>

        <section className="grid gap-5 md:grid-cols-3">
          {[
            { icon: ScanLine, title: 'YOLOv8 Detection', text: 'Vehicle localization and ROI extraction' },
            { icon: Cpu, title: 'Hybrid Recognition', text: 'EfficientNetV2 and classical CV features' },
            { icon: Activity, title: 'Model Explainability', text: 'Confidence analysis and Grad-CAM' }
          ].map((item) => (
            <div key={item.title} className="rounded-2xl border border-white/10 bg-[#101a2b] p-5">
              <item.icon className="mb-4 text-cyan-400" size={23} />
              <h3 className="font-semibold">{item.title}</h3>
              <p className="mt-1 text-sm text-slate-500">{item.text}</p>
            </div>
          ))}
        </section>

        <section className="mt-7 grid gap-6 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-2xl border border-white/10 bg-[#101a2b] p-6">
            <div className="mb-5 flex items-center justify-between">
              <h3 className="font-semibold">Input image</h3>
              <ImageIcon size={19} className="text-slate-500" />
            </div>

            <label
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault()
                setDragging(false)
                selectFile(e.dataTransfer.files[0])
              }}
              className={`flex min-h-72 cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed transition ${
                dragging ? 'border-cyan-400 bg-cyan-400/10' : 'border-slate-700 bg-[#0b1220] hover:border-cyan-500/60'
              }`}
            >
              {preview ? (
                <img src={preview} alt="Vehicle upload preview" className="max-h-80 w-full object-contain" />
              ) : (
                <>
                  <div className="mb-4 rounded-2xl bg-cyan-400/10 p-4 text-cyan-300">
                    <Upload size={28} />
                  </div>
                  <p className="font-medium">Drop your vehicle image here</p>
                  <p className="mt-2 text-xs text-slate-500">or click to browse · JPG / PNG</p>
                </>
              )}
              <input
                type="file"
                accept="image/jpeg,image/png"
                className="hidden"
                onChange={(e) => selectFile(e.target.files[0])}
              />
            </label>

            {file && (
              <div className="mt-3 flex items-center justify-between text-xs text-slate-400">
                <span className="max-w-[75%] truncate">{file.name}</span>
                <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
              </div>
            )}

            <div className="mt-6">
              <label className="mb-2 block text-sm font-medium text-slate-300">
                Detection preprocessing
              </label>
              <select
                value={preprocessing}
                onChange={(e) => setPreprocessing(e.target.value)}
                className="w-full rounded-xl border border-white/10 bg-[#0b1220] px-4 py-3 text-sm outline-none focus:border-cyan-400"
              >
                <option value="none">Original image</option>
                <option value="gaussian">Gaussian smoothing</option>
                <option value="bilateral">Bilateral filtering</option>
                <option value="clahe">CLAHE enhancement</option>
                <option value="gamma">Gamma correction</option>
              </select>
            </div>

            <button
              onClick={recognize}
              disabled={!file || loading}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-cyan-400 px-5 py-3.5 font-bold text-slate-950 transition hover:bg-cyan-300 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {loading ? <Loader2 className="animate-spin" size={19} /> : <Sparkles size={19} />}
              {loading ? 'Analyzing vehicle...' : 'Run recognition'}
            </button>

            <button
              onClick={compareMethods}
              disabled={!file || compareLoading}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl border border-cyan-400/40 px-5 py-3 text-sm font-semibold text-cyan-300 hover:bg-cyan-400/10 disabled:opacity-40"
            >
              {compareLoading
                ? <Loader2 className="animate-spin" size={18} />
                : <Activity size={18} />}
              {compareLoading ? 'Comparing methods...' : 'Compare preprocessing methods'}
            </button>

            {file && (
              <button onClick={reset} className="mt-3 flex w-full items-center justify-center gap-2 py-2 text-sm text-slate-500 hover:text-white">
                <RotateCcw size={15} /> Reset image
              </button>
            )}

            {error && (
              <div className="mt-4 flex gap-2 rounded-xl border border-red-400/20 bg-red-400/10 p-3 text-sm text-red-300">
                <AlertTriangle size={18} className="shrink-0" /> {error}
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-white/10 bg-[#101a2b] p-6">
            <div className="mb-5 flex items-center justify-between">
              <h3 className="font-semibold">Recognition output</h3>
              <span className="rounded-full bg-slate-700/50 px-3 py-1 text-xs text-slate-400">
                {results ? 'Analysis complete' : 'Awaiting image'}
              </span>
            </div>

            {results ? (
              <div>
                {results.annotated_image && (
                  <img
                    src={imageUrl(results.annotated_image)}
                    alt="Vehicle detection output"
                    className="mb-5 max-h-64 w-full rounded-xl bg-black object-contain"
                  />
                )}
                <div className="mb-5 grid grid-cols-2 gap-3">
                  <div className="rounded-xl bg-[#0b1220] p-4">
                    <p className="text-xs text-slate-500">Vehicles detected</p>
                    <p className="mt-1 text-2xl font-bold text-cyan-300">{results.detection_count}</p>
                  </div>
                  <div className="rounded-xl bg-[#0b1220] p-4">
                    <p className="text-xs text-slate-500">Recognition engine</p>
                    <p className="mt-2 text-sm font-semibold">Hybrid AI</p>
                  </div>
                </div>

                {detections.length === 0 ? (
                  <p className="rounded-xl bg-[#0b1220] p-5 text-sm text-slate-400">
                    No supported vehicle was detected in this image.
                  </p>
                ) : detections.map((det, index) => (
                  <div key={index} className="mb-5 rounded-xl border border-white/10 bg-[#0b1220] p-4">
                    <div className="mb-4 flex items-center justify-between">
                      <h4 className="font-semibold">Vehicle {index + 1}</h4>
                      <span className="text-xs text-cyan-300">
                        Detection {(det.detection_confidence * 100).toFixed(1)}%
                      </span>
                    </div>

                    {det.roi && (
                      <img src={imageUrl(det.roi)} alt="Detected vehicle ROI" className="mb-4 max-h-44 w-full rounded-lg object-contain" />
                    )}

                    <p className="mb-3 text-xs uppercase tracking-wider text-slate-500">
                      Top vehicle predictions
                    </p>

                    {det.predictions?.map((prediction, i) => (
                      <div key={prediction.class_id} className="mb-3">
                        <div className="mb-1 flex justify-between gap-3 text-sm">
                          <span className="font-medium">{i + 1}. {prediction.vehicle_model}</span>
                          <span className="shrink-0 text-cyan-300">{(prediction.confidence * 100).toFixed(2)}%</span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                          <div
                            className="h-full rounded-full bg-cyan-400"
                            style={{ width: `${Math.min(prediction.confidence * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                    ))}

                    {det.predictions?.[0] && (
                      <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-400/20 bg-amber-400/5 p-3 text-xs leading-5 text-amber-200">
                        <AlertTriangle size={16} className="mt-0.5 shrink-0" />
                        Predictions are model estimates. Verify low-confidence results against ground truth.
                      </div>
                    )}

                    {det.gradcam && (
                      <div className="mt-5">
                        <p className="mb-2 text-sm font-medium">Grad-CAM explanation</p>
                        <img src={imageUrl(det.gradcam)} alt="Grad-CAM explanation" className="max-h-56 w-full rounded-lg object-contain" />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex min-h-80 flex-col items-center justify-center rounded-xl bg-[#0b1220] text-center">
                <div className="mb-4 rounded-2xl bg-slate-800 p-5 text-slate-500">
                  <Layers3 size={34} />
                </div>
                <p className="font-medium text-slate-300">Your analysis will appear here</p>
                <p className="mt-2 max-w-xs text-sm leading-6 text-slate-500">
                  Upload an image and run recognition to view detected vehicles and model predictions.
                </p>
              </div>
            )}
          </div>
        </section>

        {comparison && (
          <section className="mt-8 rounded-2xl border border-white/10 bg-[#101a2b] p-6">
            <h3 className="mb-2 text-xl font-bold">
              Preprocessing Comparison
            </h3>
            <p className="mb-6 text-sm text-slate-400">
              Same vehicle bounding boxes, four preprocessing methods,
              and the same trained hybrid classifier.
            </p>

            {comparison.length === 0 && (
              <p className="text-slate-400">No supported vehicles found.</p>
            )}

            {comparison.map((vehicle) => (
              <div key={vehicle.detection_index} className="mb-7">
                <h4 className="mb-4 font-semibold">
                  Vehicle {vehicle.detection_index}
                  <span className="ml-2 text-sm text-cyan-300">
                    Detection {(vehicle.detection_confidence * 100).toFixed(1)}%
                  </span>
                </h4>

                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                  {['original', 'gaussian', 'bilateral', 'median'].map((method) => {
                    const result = vehicle.methods[method]
                    if (!result) return null

                    return (
                      <div key={method} className="rounded-xl border border-white/10 bg-[#0b1220] p-3">
                        <h5 className="mb-3 font-semibold capitalize text-cyan-300">
                          {method}
                        </h5>

                        {result.roi && (
                          <img
                            src={imageUrl(result.roi)}
                            alt={`${method} vehicle ROI`}
                            className="mb-3 h-32 w-full rounded-lg object-contain"
                          />
                        )}

                        <p className="text-sm font-medium">
                          {result.vehicle_model}
                        </p>

                        <p className="mt-2 text-lg font-bold text-cyan-300">
                          {(result.confidence * 100).toFixed(2)}%
                        </p>

                        <p className="text-xs text-slate-500">
                          Top-2: {result.top2_model}
                        </p>

                        <p className="mt-1 text-xs text-slate-500">
                          Confidence gap: {(result.confidence_gap * 100).toFixed(2)}%
                        </p>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </section>
        )}

        <footer className="mt-9 flex flex-col justify-between gap-3 border-t border-white/10 py-6 text-xs text-slate-600 sm:flex-row">
          <span>AutoVision AI · B.Tech Project P5</span>
          <span>YOLOv8 · EfficientNetV2 · Classical CV · Hybrid Fusion</span>
        </footer>
      </main>
    </div>
  )
}

export default App
