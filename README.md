# AI_project_

Anime genre classifier with a Next.js frontend and a FastAPI + PyTorch backend.

## What is included

- Next.js app in `app/` and `components/`
- FastAPI backend in `backend/`
- Trained model checkpoint and thresholds tracked with Git LFS in `backend/`

## Clone and set up

1. Install Git LFS before cloning if you do not already have it.
2. Clone the repository.
3. Install the frontend dependencies with `npm install` in the repo root.
4. Create and use a Python 3.12 virtual environment inside `backend/`.
5. Install backend dependencies with `pip install -r requirements.txt` from the `backend/` folder.

## Run locally

### Frontend

Create a `.env.local` file in the repo root:

```bash
PYTHON_BACKEND_URL=http://127.0.0.1:8000/classify
```

Then start the app from the repo root:

```bash
npm run dev
```

### Backend

From `backend/`, start the API:

```bash
uvicorn main:app --reload --port 8000
```

The backend expects these files in `backend/`:

- `best_phase2.pt`
- `best_thresholds_cv.npy`

If you are missing them, the backend will still start, but inference quality will be reduced.

## Notes for teammates

- The uploaded model file is large, so it is stored with Git LFS.
- Do not commit `.venv/`, `node_modules/`, or `.env.local`.
- Backend setup details are also documented in `backend/README.md`.