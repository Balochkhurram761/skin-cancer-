import express from 'express'
import dotenv from  'dotenv'
import router from './AuthRouter/authRouter.js';



dotenv.config();
// app.use(express.json());

const app=express();
app.use("/api/auth", router)

const port = process.env.PORT || 5000


app.listen(port,()=>{
    console.log(`the port server is ${port}`)
})