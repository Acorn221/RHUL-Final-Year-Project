/* eslint-disable jsx-a11y/label-has-associated-control */
import '@/index.css';
import {
  BrowserRouter as Router,
  Routes,
  Route,
} from 'react-router-dom';

/**
 * Importing the different local components
 */
import HomePage from '@/misc/HomePage';
import Alzheimers from '@/Alzheimers';
import Layout from '@/misc/Layout';
import SkinCancer from '@/SkinCancer';

/**
 * This is the main component of the application, it does the routing between the different pages
 * This allows me to have multiple pages in the same application without having to reload the page
 * @returns The main component of the application
 */
const App = () => (
  <Router>
    <Routes>
      <Route element={<Layout />}>
        <Route path="/AD-Predicter" element={<Alzheimers />} />
        <Route path="/Skin-Cancer-Predicter" element={<SkinCancer />} />
        <Route path="/" element={<HomePage />} />
        <Route path="*">404 Not Found</Route>
      </Route>
    </Routes>
  </Router>
);

export default App;
