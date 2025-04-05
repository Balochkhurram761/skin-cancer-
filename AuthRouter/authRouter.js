import express from 'express'
import upload from '../Multer/multer.js';
import { imageUpload } from '../AuthController/authController.js';


const router= express.Router();


router.post("/predict",upload.single('image'), imageUpload)



export default router;