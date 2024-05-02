import '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-converter';
// Register WebGL backend.
import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow-models/body-segmentation';
import multer from 'multer';
import bodyParser from 'body-parser';
import express from 'express';

import * as depthEstimation from '@tensorflow-models/depth-estimation';
// import * as tf from '@tensorflow/tfjs-node';
import * as tf from '@tensorflow/tfjs-node-gpu';


const app = express();
const port = 9302; 
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(express.json());

const estimationConfig = {
    minDepth: 0.2,
    maxDepth: 0.9,
};


async function runDepthEstinit() {

    const model = depthEstimation.SupportedModels.ARPortraitDepth;
    const estimatorConfig = {
        outputDepthRange: [0, 1]
    };
    const estimator = await depthEstimation.createEstimator(model, estimatorConfig);
    // console.log(estimator);
    return estimator;
}

async function runDepthEstimation(image, estimator) {
    
    //console.time('Depth Estimation Time');
    const depthMap = await estimator.estimateDepth(image, estimationConfig);
    //console.timeEnd('Depth Estimation Time');
    // console.log(depthMap.depthTensor.shape);
    let res_arr = depthMap.depthTensor.array();
    depthMap.depthTensor.dispose();
    return res_arr;
}
const estimator = await runDepthEstinit();
app.post('/estimate-depth', upload.single('imageData'), async (req, res) => {
    try {
        //const imageData = req.body.imageData;
        const imgBuffer = req.file.buffer;

        // console.time('Decode Image');
        const tensor = tf.node.decodeImage(imgBuffer);
        // console.timeEnd('Decode Image');
        
        // console.time('Depth estimation');
        var depthArray = await runDepthEstimation(tensor, estimator);
        // console.timeEnd('Depth estimation');
        
        // // console.time('Return Depth'); 
        // res.json({ depthArray });
        // // console.timeEnd('Return Depth');

        // console.time('Return Depth');
        depthArray = new Float32Array(depthArray.flat())
        res.send(Buffer.from(depthArray.buffer));
        // console.timeEnd('Return Depth');

        tensor.dispose()
        
        // console.log(tf.memory());
    } catch (error) {
        res.status(500).send(error.toString());
    }
});

app.listen(port, () => {
    console.log(`Depth estimation server listening at http://localhost:${port}`);
});