from insurance_management.bkp.vision_agent_tool import analyze_claim_image
import json
from pathlib import Path

def test_image_authenticity():
    # Current file directory
    current_dir = Path(__file__).parent

    # Go back one folder, then into sample_images
    image_path = current_dir.parent / "sample_images" / "damage1.jpeg"

    result = analyze_claim_image(image_path)

    print("\n=== FULL VISION OUTPUT ===")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_image_authenticity()

