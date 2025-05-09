import json
from plant_classifier import PlantClassifier
from pathlib import Path

if __name__ == "__main__":
    db_path = Path(__file__).parent.parent / "db"
    
    plant_classifier = PlantClassifier(db_path=db_path)
    
    # testing local image
    print("\n=== Testing with local image ===")
    local_image_path = "dataset/sample-images/1000_F_175913717_wh9WZV4aT5QAPnJ.jpg"
    result_local, similar_local = plant_classifier.classify_plant(local_image_path)
    print("Classification from local file:")
    print(json.dumps(result_local, indent=2))
