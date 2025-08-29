# CIFAR-10 Image Classifier Web Application

A beautiful and modern Flask web application that allows users to upload images and get AI-powered classification results using a pre-trained CIFAR-10 model.

## Features

- 🖼️ **Image Upload**: Drag & drop or click to upload images
- 🤖 **AI Classification**: Uses pre-trained CIFAR-10 model for accurate predictions
- 🎨 **Modern UI**: Beautiful, responsive design with gradient backgrounds
- 📱 **Mobile Friendly**: Responsive design that works on all devices
- ⚡ **Real-time Processing**: Instant classification results
- 🔒 **File Validation**: Supports PNG, JPG, JPEG, GIF, BMP formats (max 16MB)
- 📊 **Top 3 Predictions**: Shows confidence scores for the best matches

## CIFAR-10 Classes

The model can classify images into these 10 categories:
- ✈️ Airplane
- 🚗 Automobile
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐎 Horse
- 🚢 Ship
- 🚛 Truck

## Prerequisites

- Python 3.8 or higher
- Your `cifar10_advanced_model.h5` file in the project root
- At least 2GB RAM (for TensorFlow model loading)

## Installation

1. **Clone or download this project** to your local machine

2. **Navigate to the project directory**:
   ```bash
   cd "ciraf model"
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

4. **Activate the virtual environment**:
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source venv/bin/activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the application**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Upload an image**:
   - Drag and drop an image onto the upload area, or
   - Click the upload area to browse and select a file

4. **View results**: The application will display:
   - Your uploaded image
   - Top 3 classification predictions with confidence scores
   - Percentage confidence for each prediction

## Project Structure

```
ciraf model/
├── app.py                          # Main Flask application
├── cifar10_advanced_model.h5      # Your pre-trained model
├── requirements.txt                # Python dependencies
├── README.md                      # This file
├── templates/                     # HTML templates
│   └── index.html                # Main web interface
└── uploads/                      # Temporary upload folder (auto-created)
```

## API Endpoints

- `GET /` - Main application interface
- `POST /upload` - Image upload and classification endpoint
- `GET /health` - Health check endpoint

## Configuration

You can modify these settings in `app.py`:

- **File size limit**: Currently set to 16MB
- **Allowed file types**: PNG, JPG, JPEG, GIF, BMP
- **Model path**: Points to `cifar10_advanced_model.h5`
- **Port**: Default is 5000

## Troubleshooting

### Common Issues

1. **Model loading error**:
   - Ensure `cifar10_advanced_model.h5` is in the project root
   - Check that the file is not corrupted
   - Verify you have enough RAM (at least 2GB free)

2. **Import errors**:
   - Make sure you're in the virtual environment
   - Reinstall requirements: `pip install -r requirements.txt`

3. **Port already in use**:
   - Change the port in `app.py` or kill the process using port 5000

4. **Memory issues**:
   - Close other applications to free up RAM
   - The model requires significant memory to load

### Performance Tips

- **First load**: The model takes time to load initially
- **Subsequent requests**: Much faster after initial load
- **Image size**: Larger images are automatically resized to 32x32 for the model

## Deployment

For production deployment, consider:

1. **Use Gunicorn** (included in requirements):
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Set environment variables**:
   ```bash
   export FLASK_ENV=production
   export FLASK_DEBUG=0
   ```

3. **Use a reverse proxy** (Nginx/Apache) for better performance

## Security Notes

- The application includes basic file type validation
- File size limits prevent large file uploads
- Temporary files are automatically cleaned up
- Consider adding authentication for production use

## Contributing

Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve the UI/UX

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Verify your model file is valid
3. Ensure all dependencies are properly installed
4. Check the console output for error messages

---

**Happy classifying! 🎉**
