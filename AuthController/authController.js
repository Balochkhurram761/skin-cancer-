import * as tf from '@tensorflow/tfjs-node-gpu';
import fs from 'fs';
import path from 'path';
import { dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));

let model;

// Function to Load the Model (Runs Once)
async function loadModel() {
    try {
        model = await tf.loadLayersModel('file://models/model.json');
        console.log("✅ Model loaded successfully!");
    } catch (error) {
        console.error("❌ Error loading model:", error);
    }
}

// Function to Preprocess Image Before Prediction
const preprocessImage = (filePath) => {
    const imageBuffer = fs.readFileSync(filePath);
    return tf.node.decodeImage(imageBuffer, 3)
        .resizeNearestNeighbor([64, 64])
        .toFloat()
        .expandDims()
        .div(255.0);
};

// API Function to Handle Image Upload and Prediction
export const imageUpload = async (req, res) => {
    if (!req.file) {
        return res.status(400).send({
            success: false,
            message: "No image uploaded",
        });
    }

    try {
        const filePath = path.join(__dirname, req.file.path);

        // Ensure Model is Loaded Before Predicting
        if (!model) {
            return res.status(500).json({
                success: false,
                message: "Model is not loaded yet. Please try again later.",
            });
        }

        // Preprocess Image and Get Prediction
        const imageTensor = preprocessImage(filePath);
        const prediction = model.predict(imageTensor);
        const result = prediction.arraySync()[0];

        // Delete Image After Prediction
        fs.unlinkSync(filePath);

        // Send Response
        res.json({
            success: true,
            prediction: result[0] > 0.5 ? "Malignant (Cancerous)" : "Benign (Non-Cancerous)",
            confidence: result[0],
        });
    } catch (err) {
        console.error("❌ Error processing image:", err);
        res.status(500).json({
            success: false,
            message: "Error processing image",
        });
    }
};

// Load Model When Server Starts
loadModel();
