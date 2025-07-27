import NewCardInfo from "../NewCardInfo";
import Card from "./Card";
import { Link, useNavigate, useParams} from "react-router-dom";
import React from 'react';
import { useEffect } from "react";
import { useState } from "react";
import { GiShoppingCart } from "react-icons/gi";




const ProductView = () =>{
  const [click,setClick]=useState(0);

  const navigate = useNavigate();
  const callnav =() =>{
    navigate('/cart');
  }
  const goToBag =()=>{
   setClick(prevClick => prevClick + 1);
   console.log(click);
   document.getElementById("cartBTN");
   const current = cartBTN.innerHTML= 'Go to Bag';
   if(click==1){
     callnav();
   };

  }
  
  const [count,setCount]=useState(1);
  const kamkar = () =>{
    if(count > 1) {
      setCount(prevCount => prevCount - 1);
      };
    };
  let Bahda = () =>{
    if(count<10){
      setCount(prevCount => prevCount + 1)
      }
    }

    const { id } = useParams();
  const [card, setCard] = useState(null);
  
  


  useEffect(() => {
    const matchedCard = NewCardInfo.find((c) => c.id === parseInt(id, 10));

    setCard(matchedCard);
  }, [id]);

  return (
    <div>
      {card ? (
        <div className="Mainpg" >
     <div className="Topsecpg">
     
        <div className="Imagesec">
            <img src={card.image}></img>
            </div>
      
       <div className="NxtotImg">
        <div className="Parrleltoimg">

            <div className="Headofpg"><h2>{card.name}</h2></div>
            <hr />
            <div className="Lwrpadd">
            <div className="Lwrupflxsec">
            
            <div className="CupCat"><label>Category</label><h5>{card.category}</h5></div>
            
            <div className="materialsec"> <label>Material</label><h5>{card.material}</h5></div>
            <div className="Quantssec"><label>Set Of</label><h5>{card.quant}</h5></div>
            <div className="Starssec"><label>User Ratings</label><h5>{card.stars}</h5></div>

            </div>
            <hr />
            <label>Price</label>
            <div className="Pricesec"><h2>{card.price}</h2></div>
            <hr />
            <div className="Offersec">
            <label>OFFERS & DISCOUNTS</label>
            <div className="checkpincodeop">
                <input type="text" placeholder="Add Promo"></input>
                <button className="CheckPinBtn">Check</button>
            </div>
            <div className="Offrlist">
            <div className="Listofr1">
              <div id="offrbtncss" className="offrbtn"><button>{card.offrnm1}</button></div>
              <div id="offdetcss" className="offrdetail"><h6>{card.offrdtl1}</h6></div>
              </div>
              <div className="Listofr1">
              <div id="offrbtncss" className="offrbtn"><button>{card.offrnm2}</button></div>
              <div id="offdetcss" className="offrdetail"><h6>{card.offrdtl2}</h6></div>
              </div>
            </div>
            <hr />
            </div>
           <div className="Deloptions">
               <label>Delivery options</label>
               <div className="delopt1"><h5>{card.delopt1}</h5></div>
               <div className="delopt2"><h5>{card.delopt2}</h5></div>
           </div>

    
            </div>
        
        </div>
        <hr/>
        <div className="Bottomsecpg">
        
         
         <label>Check Pincode</label>
        
         <div className="Onetwobagsec">
         <div className="Pinbox">
            <div className="checkpincodeop">
                <input type="text" placeholder="Enter Pincode"></input>
                <button className="CheckPinBtn">Check</button>
            </div>
            </div>
            
            <div className="AddTObGBtn">
                <div className="counter">
                  <button onClick={kamkar}>-</button><h4>{count}</h4><button onClick={Bahda}>+</button>
                </div>
                <button id="cartBTN" onClick={goToBag}>Add to Bag <GiShoppingCart /></button>
            </div>
            </div>
            <div className="Bottomproductdesc">{card.desc}<p>
            </p>
            
            </div>
            </div>
            </div>
        </div>
        <hr/>
        <label>More info about the product</label>
        <div className="Bottomproductdesc">{card. detaileddesc}<p>
            </p>
        </div>
        <hr/>
    </div>
      ) : (
        <p>Card not found</p>
      )}
    </div>
  );
    
;
   
}

export default ProductView;