import json
import os

def labelme_to_yolo_tracking(json_dir, output_dir, img_w=1280, img_h=720):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Species mapping
    class_map = {"F1": 0, "F2": 1, "F3": 2, "F4": 3, "F5": 4, "F6": 5}

    if not os.path.exists(json_dir):
        print(f"Error: Indha path illai -> {json_dir}")
        return

    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'): continue
        
        with open(os.path.join(json_dir, json_file)) as f:
            data = json.load(f)
        
        yolo_lines = []
        counts = {} 

        for shape in data['shapes']:
            label = shape['label']
            if label.startswith('C'): continue # Corals skip
            
            if label in class_map:
                class_id = class_map[label]
                if label not in counts: counts[label] = 0
                track_id = counts[label]
                counts[label] += 1
                
                (x1, y1), (x2, y2) = shape['points']
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                w_norm, h_norm = w / img_w, h / img_h
                
                yolo_lines.append(f"{class_id} {x_center} {y_center} {w_norm} {h_norm}")

        with open(os.path.join(output_dir, json_file.replace('.json', '.txt')), 'w') as f:
            f.write("\n".join(yolo_lines))

# Neenga sonna sariyana path-ai inge kuduthuruken
train_in = r'D:\VV\dataset\labels\train_jsons'
train_out = r'D:\VV\dataset\labels\train'

val_in = r'D:\VV\dataset\labels\val_jsons'
val_out = r'D:\VV\dataset\labels\val'

print("Starting Conversion...")
labelme_to_yolo_tracking(train_in, train_out)
labelme_to_yolo_tracking(val_in, val_out)
print("Conversion Over! 'labels/train' matrum 'labels/val' check pannunga.")