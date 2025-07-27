import { MdOutlineShoppingCart } from "react-icons/md";
import { RiHeart3Line } from "react-icons/ri";
import CatagoriesCardInfo from "../CatagoriesCardInfo";
import React from 'react'
import { useNavigate } from "react-router";
import { Link } from "react-router-dom";



const CatCardLayout = ({data}) =>{
  const navigate = useNavigate();
     const gotoCart = () =>{
      navigate('/cart');
     }
    return(

     data.map((values)=>{
        const {id,price,image,name}=values;
        return <>
        
          <div className="Card1over" key={id}>
          <Link style={{textDecoration: 'none',color:"black"}} to={`/productviewcat/${id}`}>
      <div className="CardUpr">
        <img className="CardUprimg" src={image}></img>
    
      </div>
      
    
      <div className="CardDwnCont">
        <h2 className="CardTtldwn">{name}</h2>
        <h5 className="CardPricedwn">{price}</h5>
      </div></Link>
      <div className="BtnSection">
        <button onClick={gotoCart} className="AddtocardBtn"><MdOutlineShoppingCart /> Add to cart</button>
      </div>
      <Link style={{textDecoration: 'none',color:"black"}} to={`/productviewcat/${id}`}></Link>
      </div>
      
        </>

      })
      
    );
}
export default CatCardLayout;