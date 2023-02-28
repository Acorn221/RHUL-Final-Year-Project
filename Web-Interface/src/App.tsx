import React, { useEffect, useState } from 'react';

import '@/index.css';

enum states {
  waitingForFile,
  waitingForPrediction,
  predictionReceived,
}

const App = () => {
  const [currentState, setCurrentState] = useState<states>(states.waitingForFile);
  // blob or undefined
  const [image, setImage] = useState<Blob | undefined>(undefined);

  const imageUploaded = (event) => {
    setImage(event.target.files[0]);
    setCurrentState(states.waitingForPrediction);
  };

  useEffect(() => {
    if (currentState === states.waitingForPrediction) {
      const formData = new FormData();
      if (image === undefined) return;
      formData.append('file', image);
      fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data);
          setCurrentState(states.predictionReceived);
        });
    }
  }, [currentState]);

  return (
    <div className="flex justify-center align-middle h-screen">
      <div className="bg-white m-auto p-10 rounded-xl w-3/4 md:w-1/2 text-center">
        <div className="underline text-5xl">Alzheimer&apos;s Disease Predictor</div>
        <div className="m-5 text-left">
          <div className="text-2xl p-20 text-center bg-slate-300 hover:bg-slate-400 rounded-2xl select-none">
            {states.waitingForFile === currentState && (
            <div>
              <input type="file" name="file" onChange={imageUploaded} />
              <br />
              Drag and drop your MRI image here, or click to select the file
            </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
