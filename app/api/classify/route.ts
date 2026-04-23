import { NextRequest, NextResponse } from "next/server"

const VALID_GENRES = [
  "Action",
  "Adventure",
  "Avant Garde",
  "Comedy",
  "Drama",
  "Ecchi",
  "Fantasy",
  "Horror",
  "Mecha",
  "Mystery",
  "Romance",
  "Sci-Fi",
  "Slice of Life",
  "Sports",
  "Supernatural",
]

interface GenreResult {
  name: string
  confidence: number
}

interface BackendResponse {
  genres: GenreResult[]
}

export async function POST(request: NextRequest) {
  try {
    const backendUrl = process.env.PYTHON_BACKEND_URL

    if (!backendUrl) {
      return NextResponse.json(
        { error: "Backend URL not configured" },
        { status: 500 }
      )
    }

    const formData = await request.formData()
    const file = formData.get("image") as File | null

    if (!file) {
      return NextResponse.json(
        { error: "No image file provided" },
        { status: 400 }
      )
    }

    // Validate file type
    if (!["image/jpeg", "image/png"].includes(file.type)) {
      return NextResponse.json(
        { error: "Only JPG and PNG files are accepted" },
        { status: 400 }
      )
    }

    // Validate file size (5MB)
    if (file.size > 5 * 1024 * 1024) {
      return NextResponse.json(
        { error: "File size must be less than 5MB" },
        { status: 400 }
      )
    }

    // Forward the image to the Python backend
    const backendFormData = new FormData()
    backendFormData.append("file", file)

    const backendResponse = await fetch(backendUrl, {
      method: "POST",
      body: backendFormData,
    })

    if (!backendResponse.ok) {
      const errorText = await backendResponse.text()
      console.error("Backend error:", errorText)
      return NextResponse.json(
        { error: "Failed to classify image" },
        { status: backendResponse.status }
      )
    }

    const data: BackendResponse = await backendResponse.json()

    // Validate the response format
    if (!data.genres || !Array.isArray(data.genres)) {
      return NextResponse.json(
        { error: "Invalid response from classification service" },
        { status: 502 }
      )
    }

    // Filter to only valid genres and ensure confidence is a number
    const validatedGenres = data.genres
      .filter((g) => VALID_GENRES.includes(g.name))
      .map((g) => ({
        name: g.name,
        confidence: typeof g.confidence === "number" ? g.confidence : 0,
      }))

    return NextResponse.json({ genres: validatedGenres })
  } catch (error) {
    console.error("Classification error:", error)
    return NextResponse.json(
      { error: "An unexpected error occurred" },
      { status: 500 }
    )
  }
}
