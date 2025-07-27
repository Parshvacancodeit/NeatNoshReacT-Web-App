
import '../App.css'
import Navbar from './Navbar';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.min.js';
import LandPage from './LandPage';
import Card from './Card';
import NewLaunchPage from './NewLaunchPage';
import SalePage from './SalePage';
import AutoSlider from './AutoSlider';

import Combinercat from './Combinercat';
import Footer from './Footer';
import { BrowserRouter } from 'react-router-dom';
import Bloggspot from './Bloggspot';
const Home = () => {



    return (
         <>
   
      <SalePage />
      <LandPage />
      <NewLaunchPage />
      <div className='ApnaKard'>
      <Card />
      </div>

      <Combinercat />
      <Bloggspot />
      <AutoSlider />
      <Footer />
      </>
    );

}

export default Home;