"use client"

import { useState } from "react"
import { UploadCard } from "@/components/upload-card"
import { ResultsPanel } from "@/components/results-panel"

interface Genre {
  name: string
  confidence: number
}

export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState<Genre[] | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleClassify = async (file: File) => {
    setIsLoading(true)
    setError(null)

    // Create preview for results
    const reader = new FileReader()
    reader.onload = (e) => {
      setImagePreview(e.target?.result as string)
    }
    reader.readAsDataURL(file)

    try {
      const formData = new FormData()
      formData.append("image", file)

      const response = await fetch("/api/classify", {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        const data = await response.json()
        throw new Error(data.error || "Failed to classify image")
      }

      const data = await response.json()
      setResults(data.genres)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred")
      setResults(null)
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setResults(null)
    setImagePreview(null)
    setError(null)
  }

  return (
    <main className="min-h-screen bg-slate-900 py-12 px-4">
      {/* Hero Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl md:text-5xl font-bold text-white mb-4">
          Anime Genre Classifier
        </h1>
        <p className="text-slate-400 text-lg max-w-xl mx-auto">
          Upload an anime cover image and discover its genres using AI-powered classification
        </p>
      </div>

      {/* Upload Card */}
      {!results && (
        <UploadCard onClassify={handleClassify} isLoading={isLoading} />
      )}

      {/* Error Message */}
      {error && !results && (
        <div className="w-full max-w-lg mx-auto mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
          <p className="text-red-400 text-center">{error}</p>
        </div>
      )}

      {/* Results Panel */}
      {results && imagePreview && (
        <ResultsPanel
          imagePreview={imagePreview}
          genres={results}
          onReset={handleReset}
        />
      )}
    </main>
  )
}
