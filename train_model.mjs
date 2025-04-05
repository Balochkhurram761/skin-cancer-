import fs from 'fs'
import *  as tf from '@tensorflow/tfjs-node'

const IMAGE_SIZE=64

const TRAING_DATA=`dataset/train`


const loadImageFromFolder=async(folder)=>{
    const files =fs.readdirSync(folder)    //Folder k ander jitni files hain subfolder hain  in ko ready karna  or array ma dhkna hain
    const images=[];

    for( const file of files){

        const filepath= `${folder}/${file}`;
        const imageBuffer=fs.readFileSync(filepath)
        const tensor =tf.node.decodeImage(imageBuffer ,3)
        .resizeNearestNeighbor([IMAGE_SIZE, IMAGE_SIZE])    // 300 x 300 pixel karta etc
        .toFloat()   // point ma convert karta hain
        .expandDims(0)  // Tensor mein ek extra dimension add karta hai, jo batch dimension hota hai. (1,255,255 ,3)
        .div(255.0)       //Har pixel value ko 255.0 se divide karta hai.
        images.push(tensor)
    }
    return images;
}

const createmodel=()=>{
    const model=tf.sequential();   // Simple aur Linear Model Banane ke Liye
 
 
    model.add(tf.layers.conv2d({ 
      filters:32,             //Filters: Features detect karte hain (edges, textures, etc.).  32 k mtlb hain k 32 apply hoga etc
     kernelSize:3,          //Kernel (3x3): Image ke small regions par kaam karta hai.
      activation:`relu`,     //ReLU: Negative values ko 0 banata hai. example -3 hain tu is ko 0 bana dain ga
          //Output Shape: Tumhara image depth badh jata hai filters ke count ke according.
    }));
 model.add(tf.layers.maxPool2d({
    poolSize:2      //Max pooling ka kaam hai size reduce karna aur strong features retain karna. 
                   // poolSize: 2 ka matlab hai har 2x2 ka block consider karega, aur maximum value select karega.
 }))



model.add(tf.layers.conv2d({ 
    filters:64,             //Filters: Features detect karte hain (edges, textures, etc.).  32 k mtlb hain k 32 apply hoga etc
   kernelSize:3,          //Kernel (3x3): Image ke small regions par kaam karta hai.
    activation:`relu`,     //ReLU: Negative values ko 0 banata hai. example -3 hain tu is ko 0 bana dain ga
   inputShape:[IMAGE_SIZE,IMAGE_SIZE, 3],    //Output Shape: Tumhara image depth badh jata hai filters ke count ke according.
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

return model;



}



const trainModel=async()=>{
 const beginImages=await loadImageFromFolder(`${TRAING_DATA}/benign`);
 const malignantImages=await loadImageFromFolder(`${TRAING_DATA}/malignant`)
 const x=tf.concat([...beginImages,...malignantImages])
 const y= tf.tensor1d([
   ...Array(beginImages.length).fill(0),
   ...Array(malignantImages.length).fill(1)
 ])
 const model=createmodel();
 await model.fit(x,y,{epochs:10 , batchSize:32});
 await model.save('file://models');   //Ye trained model ko save karne ke liye use hota hai. Jab aap model ko train kar lete hain, toh usko future me use karne ke liye save kar sakte hain, taake dobara training na karni pade.


}