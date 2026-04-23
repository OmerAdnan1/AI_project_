"use client"

import { useCallback, useState, useRef } from "react"
import { Upload, X, ImageIcon } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Spinner } from "@/components/ui/spinner"

interface UploadCardProps {
  onClassify: (file: File) => void
  isLoading: boolean
}

const MAX_FILE_SIZE = 5 * 1024 * 1024 // 5MB
const ACCEPTED_TYPES = ["image/jpeg", "image/png"]

export function UploadCard({ onClassify, isLoading }: UploadCardProps) {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const validateFile = (file: File): string | null => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      return "Only JPG and PNG files are accepted"
    }
    if (file.size > MAX_FILE_SIZE) {
      return "File size must be less than 5MB"
    }
    return null
  }

  const handleFile = useCallback((selectedFile: File) => {
    const validationError = validateFile(selectedFile)
    if (validationError) {
      setError(validationError)
      return
    }
    
    setError(null)
    setFile(selectedFile)
    
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target?.result as string)
    }
    reader.readAsDataURL(selectedFile)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile) {
      handleFile(droppedFile)
    }
  }, [handleFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) {
      handleFile(selectedFile)
    }
  }

  const handleRemove = () => {
    setFile(null)
    setPreview(null)
    setError(null)
    if (inputRef.current) {
      inputRef.current.value = ""
    }
  }

  const handleClassify = () => {
    if (file) {
      onClassify(file)
    }
  }

  return (
    <div className="w-full max-w-lg mx-auto bg-slate-800 rounded-xl border border-slate-700 p-6 shadow-xl">
      {!preview ? (
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`
            relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200
            ${isDragging 
              ? "border-violet-500 bg-violet-500/10" 
              : "border-slate-600 hover:border-slate-500"
            }
          `}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png"
            onChange={handleInputChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-full bg-slate-700 flex items-center justify-center">
              <Upload className="w-8 h-8 text-violet-500" />
            </div>
            <div>
              <p className="text-slate-200 font-medium">
                Drag and drop your anime cover here
              </p>
              <p className="text-slate-400 text-sm mt-1">
                or click to browse
              </p>
            </div>
            <p className="text-slate-500 text-xs">
              JPG or PNG, max 5MB
            </p>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="relative">
            <img
              src={preview}
              alt="Preview"
              className="w-full h-64 object-contain rounded-lg bg-slate-900"
            />
            <button
              onClick={handleRemove}
              disabled={isLoading}
              className="absolute top-2 right-2 w-8 h-8 rounded-full bg-slate-900/80 hover:bg-slate-900 flex items-center justify-center transition-colors disabled:opacity-50"
            >
              <X className="w-4 h-4 text-slate-300" />
            </button>
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <ImageIcon className="w-4 h-4" />
            <span className="truncate">{file?.name}</span>
            <span className="text-slate-500">
              ({(file?.size ? file.size / 1024 : 0).toFixed(1)} KB)
            </span>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      <Button
        onClick={handleClassify}
        disabled={!file || isLoading}
        className="w-full mt-6 bg-violet-500 hover:bg-violet-600 text-white font-medium py-3 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <span className="flex items-center justify-center gap-2">
            <Spinner className="w-4 h-4" />
            Analysing cover...
          </span>
        ) : (
          "Classify"
        )}
      </Button>
    </div>
  )
}
