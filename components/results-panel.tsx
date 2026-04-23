"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { RotateCcw } from "lucide-react"

interface Genre {
  name: string
  confidence: number
}

interface ResultsPanelProps {
  imagePreview: string
  genres: Genre[]
  onReset: () => void
}

const CONFIDENCE_THRESHOLD = 0.15

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.6) return "bg-green-500"
  if (confidence >= 0.4) return "bg-amber-500"
  return "bg-red-500"
}

function getConfidenceBgColor(confidence: number): string {
  if (confidence >= 0.6) return "bg-green-500/20"
  if (confidence >= 0.4) return "bg-amber-500/20"
  return "bg-red-500/20"
}

export function ResultsPanel({ imagePreview, genres, onReset }: ResultsPanelProps) {
  const filteredGenres = genres
    .filter((g) => g.confidence >= CONFIDENCE_THRESHOLD)
    .sort((a, b) => b.confidence - a.confidence)

  if (filteredGenres.length === 0) {
    return (
      <div className="w-full max-w-2xl mx-auto bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-xl animate-in fade-in slide-in-from-bottom-4 duration-500">
        <p className="text-slate-400 text-center">
          No genres detected above the confidence threshold.
        </p>
        <Button
          onClick={onReset}
          variant="outline"
          className="w-full mt-4 border-slate-600 text-slate-300 hover:bg-slate-700"
        >
          <RotateCcw className="w-4 h-4 mr-2" />
          Classify another
        </Button>
      </div>
    )
  }

  return (
    <div className="w-full max-w-2xl mx-auto bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-xl animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex flex-col md:flex-row gap-6">
        {/* Image Thumbnail */}
        <div className="flex-shrink-0">
          <img
            src={imagePreview}
            alt="Classified anime cover"
            className="w-full md:w-32 h-44 object-cover rounded-lg bg-slate-900"
          />
        </div>

        {/* Genre Results */}
        <div className="flex-1 space-y-3">
          <h3 className="text-slate-200 font-semibold text-lg mb-4">
            Predicted Genres
          </h3>
          {filteredGenres.map((genre) => (
            <div key={genre.name} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-slate-300">{genre.name}</span>
                <span className="text-slate-400">
                  {(genre.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div
                className={`h-2 rounded-full ${getConfidenceBgColor(genre.confidence)}`}
              >
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getConfidenceColor(genre.confidence)}`}
                  style={{ width: `${genre.confidence * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Genre Badges */}
      <div className="mt-6 pt-4 border-t border-slate-700">
        <p className="text-slate-400 text-sm mb-3">Predicted Labels</p>
        <div className="flex flex-wrap gap-2">
          {filteredGenres.map((genre) => (
            <Badge
              key={genre.name}
              variant="secondary"
              className="bg-violet-500/20 text-violet-300 border-violet-500/30 hover:bg-violet-500/30"
            >
              {genre.name}
            </Badge>
          ))}
        </div>
      </div>

      {/* Reset Button */}
      <Button
        onClick={onReset}
        variant="outline"
        className="w-full mt-6 border-slate-600 text-slate-300 hover:bg-slate-700"
      >
        <RotateCcw className="w-4 h-4 mr-2" />
        Classify another
      </Button>
    </div>
  )
}
