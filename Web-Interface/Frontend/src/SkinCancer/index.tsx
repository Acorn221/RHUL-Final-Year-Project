import React, { useState, useEffect } from 'react';
// import tensorflow library and the type for loadGraphModel
import * as tf from '@tensorflow/tfjs';

const SkinCancer = () => {
  const [image, setImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState<tf.LayersModel | null>(null);

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
