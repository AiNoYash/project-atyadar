# AdaFace Retrieval API

This project uses AdaFace to retrieve the best-matching facial image from a database based on a generated or input image. It includes a built-in guidance system to track whether sequentially generated images are getting closer to or further from the target identity.

## Setup Instructions

### 1. Download Model Weights
You must download the pre-trained AdaFace weights before running any scripts.

*   **Download Link:** [adaface_ir50_ms1mv2.ckpt](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view)
*   **Placement:** Place the downloaded `.ckpt` file inside the **top-level** `AdaFace` folder. **Do not** put it inside the nested `AdaFace/AdaFace/` directory.

Your folder structure should look like this:
```text
IVP PROJ/
├── AdaFace/                  <-- TOP LEVEL FOLDER
│   ├── adaface_ir50_ms1mv2.ckpt  <-- PUT WEIGHTS HERE
│   ├── database_embeddings.pt
│   ├── AdaFace/              <-- NESTED FOLDER (Cloned Repo)
│   └── your_script.py
```

### 2. Install Dependencies
Make sure your virtual environment is active, then install the requirements:
```bash
pip install -r AdaFace/requirements.txt
```
*(Note: this for top level AdaFace folder do not use requirements.txt of nexted AdaFace folder).*

---

## API Usage Guide

The system is wrapped in a single, easy-to-use class: `AdaFaceMatcher`.

### Initialization
Load the matcher once when your server/application starts. This keeps the model and database embeddings in memory for fast inference.

```python
from AdaFaceMatcher import AdaFaceMatcher

matcher = AdaFaceMatcher(
    weight_path='adaface_ir50_ms1mv2.ckpt', 
    database_path='database_embeddings.pt'
)
```

### Creating the Database (Run Once)
If you don't have `database_embeddings.pt` yet, or if you add new images to your dataset folder, build the embeddings using the static method:

```python
# Pass your folder of images, the loaded model, and the desired output file name
AdaFaceMatcher.create_database_embeddings('indian_faces_50', matcher.model)
```

### Finding a Match
To process an incoming generated/test image and find its closest match in the database:

```python
best_match, similarity_score = matcher.find_best_match('path/to/test_image.jpg')

print(f"Matched Image: {best_match}")
print(f"Score: {similarity_score}") # -1.0 to 1.0 (Higher is better)
```

### Using the Guidance System
The guidance system tracks iterations to tell the user if their newly generated image is a better or worse match than their previous attempt.

1. **Turn on Guidance:**
   ```python
   matcher.guide() # Toggles guidance mode ON
   ```

2. **Run your match:**
   ```python
   matcher.find_best_match('path/to/generated_v1.jpg')
   matcher.find_best_match('path/to/generated_v2.jpg')
   ```

3. **Get Guidance Feedback:**
   ```python
   feedback = matcher.giveGuidance()
   print(feedback)
   ```

**Guidance Output Format (JSON-friendly):**
```json
{
    "improvedResults": true, 
    "targetChanged": false,
    "currentSim": 0.65,
    "prevSim": 0.52,
    "bestMatch": "person_1.jpg",
    "prevMatch": "person_1.jpg"
}
```
*   `improvedResults`: `True` if the new image looks more like the database match than the previous image.
*   `targetChanged`: `True` if the model suddenly thinks the new image looks like a completely different person in the database.