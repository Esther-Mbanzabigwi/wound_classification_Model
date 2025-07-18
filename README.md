# C-Section Wound Classification App

## Description
This project is a comprehensive wound classification system that combines a mobile application with advanced deep learning capabilities. The app allows medical professionals to capture or upload wound images and receive instant classifications of wound types (venous, diabetic, pressure, and surgical) using a trained deep learning model.

## GitHub Repository
[Link to GitHub Repository](https://github.com/Esther-Mbanzabigwi/wound_classification_Model.git)

## Environment Setup

### Python Environment (For Model Training)
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Mobile App Setup (React Native)
1. Prerequisites:
   - Node.js (v14 or higher)
   - Java Development Kit (JDK) 11
   - Android Studio with Android SDK
   - React Native CLI

2. Install dependencies:
```bash
cd WoundClassifierApp
npm install
```

3. Run the app:
```bash
# Start Metro bundler
npm start

# Run on Android
npm run android

# Build Android APK
npm run build:android
```

## Project Structure
```
F-c-section/
├── improved_wound_classification.ipynb  # Model training notebook
├── train_model.py                      # Model training script
├── wound_classifier_best.pth           # Trained model weights
├── requirements.txt                    # Python dependencies
└── WoundClassifierApp/                 # Mobile application
    ├── src/
    │   ├── components/                 # React components
    │   ├── screens/                    # App screens
    │   └── utils/                      # Utility functions
    └── android/                        # Android specific files
```

## App Interface Screenshots

[Insert screenshots of key app interfaces here]
1. Home Screen
2. Camera Screen
3. Gallery Screen
4. Results Screen

## Deployment Plan

### Mobile App Deployment
1. **Testing Phase**
   - Internal testing with development team
   - Beta testing with selected medical professionals
   - Bug fixes and performance optimization

2. **Production Release**
   - Generate signed APK
   - Upload to Google Play Store
   - Monitor crash reports and user feedback

3. **Maintenance**
   - Regular updates for bug fixes
   - Model improvements
   - Feature enhancements based on user feedback

### Model Deployment
1. **Model Optimization**
   - Convert to TorchScript format
   - Quantization for mobile deployment
   - Performance testing on target devices

2. **Integration**
   - Embed model in mobile app
   - Implement version control for model updates
   - Setup monitoring for model performance

## Video Demo
[Link to Video Demo]

The video demonstration covers:
1. App Overview
2. Image Capture Process
3. Gallery Image Selection
4. Wound Classification Process
5. Results Interpretation
6. Additional Features

## Technical Details

### Model Architecture
- Based on state-of-the-art deep learning architecture
- Trained on the AZH dataset with 730 wound images
- Supports classification of four wound types:
  - Venous
  - Diabetic
  - Pressure
  - Surgical

### Mobile App Features
- Real-time wound image capture
- Gallery image selection
- Secure data handling
- Offline classification capability
- Result sharing functionality
- Historical record keeping

## License
[Insert License Information]

## Acknowledgments
Based on research work published in:
1. Patel, Y., et al. (2024). Integrated image and location analysis for wound classification: a deep learning approach. Scientific Reports, 14(1).
2. Anisuzzaman, D. M., et al. (2022). Multi-modal wound classification using wound image and location by deep neural network. Scientific Reports, 12(1). 
