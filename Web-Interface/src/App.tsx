import React, { useState } from 'react';

import '@/index.css';

const App = () => {
  const [image, setImage] = useState({});

  return (
    <div className="flex justify-center align-middle h-screen">
      <div className="bg-white m-auto p-10 rounded-xl w-3/4 md:w-1/2 text-center">
        <div className="underline text-5xl">Alzheimer&apos;s Disease Predictor</div>
        <div className="m-5 text-left">
          <div className="text-2xl p-20 text-center bg-slate-300 hover:bg-slate-400 rounded-2xl">
            Drag and drop your MRI image here, or click to select the file
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
