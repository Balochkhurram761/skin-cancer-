import express from 'express'
import dotenv from  'dotenv'




dotenv.config();
// app.use(express.json());

const app=express();


const port = process.env.PORT || 5000


app.listen(port,()=>{
    console.log(`the port server is ${port}`)
})