from flask import Flask, request, jsonify, send_file, render_template
from PIL import Image
import io
import base64


app = Flask(__name__)

def process_images(image1, image2):
    # Open images
    Target = Image.open(image1)
    Source = Image.open(image2)
    
    # save_ source here examples/FaceSwap/One_source
    Source.save('examples/FaceSwap/One_source/source.jpg')
    
    # save_ target here examples/FaceSwap/One_target
    Target.save('examples/FaceSwap/One_target/target.jpg')
    
    # run sh file
    
    # subprocess.run(['./run.sh'])
    
    # Example processing: Combine images side by side
    new_width = Target.width + Source.width
    new_height = max(Target.height, Source.height)
    new_image = Image.new('RGB', (new_width, new_height))

    new_image.paste(Target, (0, 0))
    new_image.paste(Source, (Target.width, 0))

    # Save to a bytes buffer
    img_io = io.BytesIO()
    new_image.save(img_io, 'JPEG')
    img_io.seek(0)

    return img_io

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_images', methods=['POST'])
def process_images_endpoint():
    image1 = request.files['image1']
    image2 = request.files['image2']
    
    # Process images
    output_image_io = process_images(image1, image2)
    
    # Convert images to base64 strings
    input_image1 = base64.b64encode(image1.read()).decode('utf-8')
    input_image2 = base64.b64encode(image2.read()).decode('utf-8')
    output_image = base64.b64encode(output_image_io.getvalue()).decode('utf-8')
    
    return jsonify({
        'input_image1': f"data:image/jpeg;base64,{input_image1}",
        'input_image2': f"data:image/jpeg;base64,{input_image2}",
        'output_image': f"data:image/jpeg;base64,{output_image}"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')