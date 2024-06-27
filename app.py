import numpy as np
import torch as tc
import os
import shutil
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

def tensor_to_numpy_image(tensor):
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    numpy_image = tensor.cpu().numpy()

    if numpy_image.min() < 0 or numpy_image.max() > 255:
        numpy_image = (255 * (numpy_image - numpy_image.min()) / (numpy_image.max() - numpy_image.min())).astype(np.uint8)
    else:
        numpy_image = numpy_image.astype(np.uint8)

    return numpy_image

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("templates/index.html") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.get("/about", response_class=HTMLResponse)
async def read_about():
    with open("templates/about.html") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.get("/contact", response_class=HTMLResponse)
async def read_contact():
    with open("templates/contact.html") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.get("/project", response_class=HTMLResponse)
async def read_project():
    with open("templates/project.html") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.get("/results", response_class=HTMLResponse)
async def read_results():
    with open("templates/results.html") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.get("/upload", response_class=HTMLResponse)
async def read_upload():
    with open("templates/upload.html") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/uploadfile/")
async def create_upload_file(files: List[UploadFile] = File(...)):
    upload_directory = "static/uploads"
    display_directory = "static/uploads_display"
    os.makedirs(upload_directory, exist_ok=True)
    os.makedirs(display_directory, exist_ok=True)

    filenames = []
    for file in files:
        file_location = os.path.join(upload_directory, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        if file.filename.lower().endswith('.tif'):
            img = cv2.imread(file_location, cv2.IMREAD_UNCHANGED)
            if img is not None:
                png_filename = file.filename.rsplit('.', 1)[0] + '.png'
                png_location = os.path.join(display_directory, png_filename)
                cv2.imwrite(png_location, img)
                print(f"Converted {file.filename} to {png_filename}")
                filenames.append(png_filename)
            else:
                print(f"Failed to read uploaded image {file.filename}")
                filenames.append(file.filename)
        else:
            display_file_location = os.path.join(display_directory, file.filename)
            shutil.copy(file_location, display_file_location)
            filenames.append(file.filename)

    return {"filenames": filenames, "message": "Files uploaded successfully."}

@app.get("/uploaded_images/", response_class=JSONResponse)
async def get_uploaded_images():
    display_directory = "static/uploads_display"
    filenames = os.listdir(display_directory)
    return {"filenames": filenames}

@app.get("/results_images/", response_class=JSONResponse)
async def get_results_images():
    upload_directory = "static/uploads"
    filenames = os.listdir(upload_directory)
    return {"filenames": filenames}

@app.get("/scanned_images/", response_class=JSONResponse)
async def get_scanned_images():
    scan_directory = "static/scanned_images"
    filenames = os.listdir(scan_directory)
    return {"filenames": filenames}

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
