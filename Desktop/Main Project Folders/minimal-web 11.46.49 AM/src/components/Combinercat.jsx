
import CatagoriesCardInfo from "../CatagoriesCardInfo";
import { MdOutlineShoppingCart } from "react-icons/md";
import { useState } from "react";
import { BrowserRouter,Route,Routes} from "react-router-dom";
import CatCardLayout from "./CatCardLayout";
const Combinercat = () => {
  const  [ data , setdata] = useState(CatagoriesCardInfo);
  const filterresult = (catItems) => {
   
    console.log(catItems);
    const result = CatagoriesCardInfo.filter((p) => {
      
      return p.category===catItems
    })
    setdata(result);
    console.log(result);
  };

 
  return (

    
    <div className="PuraPuraCard">
      <div className="Catagories_sec">
        <h3 className="catagorSec">Search By Catagories</h3>
        <hr id="breakLine" />
        <ul>
        
          <button onClick={() => filterresult('Cups')}>Cups</button>
          <button onClick={() => filterresult('Dinnerware_Sets')}>
            Dinnerware Sets
          </button>
          <button onClick={() => filterresult("Bowls")}>
            Bowls
          </button>
          <button onClick={() => filterresult("Tea_Sets")}>Tea Sets</button>
          <button onClick={() => filterresult("Limited_Edition")}>
            Limited Edition
          </button>
        </ul>
      


         
    
      </div>
      <div className="TotalCards">
      <CatCardLayout data={data}/>
      </div>
    </div>
  );
      
};
export default Combinercat;
