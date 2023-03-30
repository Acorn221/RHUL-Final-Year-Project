import Button from '@mui/material/Button';

// Import react router link
import { Link } from 'react-router-dom';

const HomePage = () => (
  <div className="flex flex-col items-center justify-center h-screen">
    <div className="flex flex-col items-center justify-center m-3 max-w-4xl bg-white p-5 gap-4">
      <div className="text-4xl font-bold">Welcome to the Disease Predicter</div>
      <div className="text-2xl">This is a web application that predicts the probability of Alzheimer&apos;s disease, or predicts whether or not a skin blemish is benign or malignant</div>
      <div className="w-full justify-center flex flex-auto m-2 gap-2">
        <Link to="/AD-Predicter">
          <Button variant="contained" size="large" className="max-w-4xl h-20">Alzheimer&apos;s Predicter</Button>
        </Link>
        <Link to="/Skin-Cancer-Predicter">
          <Button variant="contained" size="large" className="max-w-4xl h-20">Skin Cancer Predicter</Button>
        </Link>
      </div>
    </div>
  </div>
);

export default HomePage;
