import fs from 'fs'
import *  as tf from '@tensorflow/tfjs-node'

const IMAGE_SIZE=64
const TRAING_DATA=`dataset/train`


const loadImageFromFolder=async(folder)=>{
    console.log(`Loading images from folder: ${folder}`);
    const files =fs.readdirSync(folder)    //Folder k ander jitni files hain subfolder hain  in ko ready karna  or array ma dhkna hain
    const images=[];

    for( const file of files){

        const filepath= `${folder}/${file}`;
        console.log(`Loading image: ${filepath}`);
        const imageBuffer=fs.readFileSync(filepath)
        const tensor =tf.node.decodeImage(imageBuffer ,3)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])    // 300 x 300 pixel karta etc
        .toFloat()   // point ma convert karta hain
        .expandDims(0)  // Tensor mein ek extra dimension add karta hai, jo batch dimension hota hai. (1,255,255 ,3)
        .div(255.0)       //Har pixel value ko 255.0 se divide karta hai.
        images.push(tensor)
       
    }
    console.log(`Loaded ${images.length} images from ${folder}`);
    return images;
}

const createmodel=()=>{
    console.log("Creating CNN model..."); 
    const model=tf.sequential();   // Simple aur Linear Model Banane ke Liye
 
 
    model.add(tf.layers.conv2d({
        filters: 32,  // Number of filters (features to detect)
        kernelSize: 3,  // Size of the convolutional kernel (3x3)
        activation: 'relu',  // Activation function (ReLU for non-linearity)
        inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],  // Specify the input shape (image size and channels)
    }));
 model.add(tf.layers.maxPool2d({
    poolSize:2      //Max pooling ka kaam hai size reduce karna aur strong features retain karna. 
                   // poolSize: 2 ka matlab hai har 2x2 ka block consider karega, aur maximum value select karega.
 }))



 model.add(tf.layers.conv2d({
    filters: 64,  // Number of filters (features to detect)
    kernelSize: 3,  // Size of the convolutional kernel (3x3)
    activation: 'relu',  // Activation function (ReLU for non-linearity)
    inputShape: [IMAGE_SIZE, IMAGE_SIZE, 3],  // Specify the input shape (image size and channels)
}));
  
  model.add(tf.layers.maxPool2d({
    poolSize:2      //Max pooling ka kaam hai size reduce karna aur strong features retain karna. 
                   // poolSize: 2 ka matlab hai har 2x2 ka block consider karega, aur maximum value select karega.
 }));
 model.add(tf.layers.flatten());  //Flatten layer kisi bhi multi-dimensional input ko ek single-dimensional vector mein convert karti hai.
 //Input Shape:  [32, 32, 64]
// Output Shape: [1, 65536] (32 x 32 x 64 = 65536)

model.add (tf.layers.dense({
    units:128,
    activation:'relu'
}))
model.add(tf.layers.dense({
    units:1,
    activation:'sigmoid'
}))
model.compile({
    optimizer:tf.train.adam(),
    loss:'binaryCrossentropy',
    metrics:['accuracy']
});
console.log("Model created successfully.");
return model;



}



const trainModel=async()=>{
    console.log("Starting model training...");
 const beginImages=await loadImageFromFolder(`${TRAING_DATA}/benign`);
 const malignantImages=await loadImageFromFolder(`${TRAING_DATA}/malignant`)
 const x=tf.concat([...beginImages,...malignantImages])
 const y= tf.tensor1d([
   ...Array(beginImages.length).fill(0),
   ...Array(malignantImages.length).fill(1)
 ])
 console.log("Training data prepared. Starting model training...");
 const model=createmodel();
 await model.fit(x, y, {
    epochs: 10,  
    batchSize: 32,  
}).then(() => {
    console.log("Model training completed.");  
    model.save('file://models');  // Save the trained model to disk
    console.log("Model saved successfully.");  
}).catch(error => {
    console.error("Error during model training:", error);  
});
 


}

trainModel();