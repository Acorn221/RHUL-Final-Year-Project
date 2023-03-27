import { useState, useEffect } from 'react';
// import tensorflow library and the type for loadGraphModel
import * as tf from '@tensorflow/tfjs';

/**
 * This is the page to predict skin cancer, it allows the user to upload an image, then processes it locally
 * to give the user a prediction of whether or not they have skin cancer
 *
 * This would be a client facing page, so the client could see if they need to speak to a doctor about their skin
 *
 * @returns The skin cancer predicter page
 */
const SkinCancer = () => {
  // This stores the image that the user uploads
  const [image, setImage] = useState(null);
  // This stores the prediction that the model gives
  const [prediction, setPrediction] = useState(null);
  // This stores the model that is loaded from the server
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  // This is to give the user feedback that the model is loading (as it might take a while)
  const [loading, setLoading] = useState(false);

  // this is the url of the model that is to be loaded
  const modelURL = `${window.location.origin}/model/model.json`;

  useEffect(() => {
    async function loadModel() {
      setLoading(true);
      const loadedModel = await tf.loadLayersModel(modelURL);
      setModel(loadedModel);
      setLoading(false);
    }
    loadModel();
  }, []);

  return (
    <div>
      <h1>Skin Cancer Predicter</h1>
      {loading && <p>Loading Model...</p>}
    </div>
  );
};

export default SkinCancer;
